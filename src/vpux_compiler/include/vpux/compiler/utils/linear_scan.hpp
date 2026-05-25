//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Linear Scan Memory Allocator
// =============================
//
// Linear Scan is a memory allocation strategy that assigns memory addresses to live ranges
// (buffers that are alive during a specific time interval). When memory is insufficient,
// it can "spill" lower-priority ranges to make room for higher-priority ones.
//
// For comprehensive examples and usage patterns, see:
//   tests/unit/vpux_compiler/utils/linear_scan_tests.cpp
//
// Handler Interface Requirements:
//
// LinearScan works in conjunction with a Handler class that implements the following
// interface methods to interact with and manage live ranges. The Handler acts as a
// policy class defining how live ranges are queried, allocated, freed, and spilled.
//
// Required methods:
//
//   * bool isAlive(LiveRange) const;
//   * bool isFixedAlloc(LiveRange) const;
//   * AddressType getSize(LiveRange) const;
//   * AddressType getAlignment(LiveRange) const;
//   * AddressType getAddress(LiveRange) const;
//   * void allocated(LiveRange, AddressType) const;
//   * void freed(LiveRange) const;
//   * <type> getSpillWeight(LiveRange) const;
//   * bool spilled(LiveRange) const;
//

#pragma once

#include "vpux/compiler/utils/partitioner.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <initializer_list>
#include <utility>

#include <cassert>

namespace vpux {

template <class LiveRange, class Handler>
class LinearScan final {
public:
    using Direction = Partitioner::Direction;
    using LiveRangeVector = SmallVector<LiveRange>;
    using LiveRangeIter = typename LiveRangeVector::iterator;
    using ReservedAddressAndSizeVector = ArrayRef<std::pair<vpux::AddressType, vpux::AddressType>>;

public:
    explicit LinearScan(AddressType size): _partitioner{size} {
    }

    template <typename... Args>
    explicit LinearScan(AddressType size, const ReservedAddressAndSizeVector& reservedVec, Args&&... args)
            : _partitioner{size}, _handler{std::forward<Args>(args)...} {
        for (const auto& addressAndSize : reservedVec) {
            _partitioner.allocFixed(addressAndSize.first, addressAndSize.second);
        }
    }

public:
    /// Frees all dead live ranges.
    /// Iterates from back to front because freeLiveRangeAt() uses swap-and-pop deletion.
    /// If we iterated forward, we would skip checking the element that was swapped from the back.
    void freeDeadRanges() {
        for (auto i = _liveRanges.size(); i > 0; --i) {
            const auto idx = i - 1;
            if (!_handler.isAlive(_liveRanges[idx])) {
                freeLiveRangeAt(idx);
            }
        }
    }

    /// Allocate memory for new live ranges, with optional spilling if space is insufficient.
    /// @param newLiveRanges Collection of live ranges to allocate memory for
    /// @param allowSpills If true, allows spilling of existing ranges when space is insufficient; if false, allocation
    /// fails when out of space
    /// @param dir Allocation direction: Direction::Up allocates from low to high addresses, Direction::Down from high
    /// to low
    /// @return true if all ranges allocated successfully, false otherwise and no changes are made
    template <class LiveRanges>
    bool alloc(const LiveRanges& newLiveRanges, bool allowSpills = true, Direction dir = Direction::Up) {
        AllocationGuard guard(_partitioner);
        SmallVector<std::pair<LiveRange, AddressType>> pendingAllocations;

        // Allocate fixed live ranges first
        for (const auto& newRange : newLiveRanges) {
            if (!_handler.isFixedAlloc(newRange)) {
                continue;
            }

            if (!allocFixedRange(newRange, allowSpills, guard)) {
                return false;
            }

            const auto addr = _handler.getAddress(newRange);
            pendingAllocations.push_back({newRange, addr});
        }

        // Allocate dynamic live ranges
        for (const auto& newRange : newLiveRanges) {
            if (_handler.isFixedAlloc(newRange)) {
                continue;
            }

            AddressType addr = InvalidAddress;
            if (!allocDynamicRange(newRange, allowSpills, dir, guard, addr)) {
                return false;
            }

            pendingAllocations.push_back({newRange, addr});
        }

        // Commit all allocations
        for (const auto& [range, addr] : pendingAllocations) {
            assert(addr != InvalidAddress);
            _handler.allocated(range, addr);
            _liveRanges.push_back(range);
        }

        guard.commit();
        return true;
    }

    /// Allocates a live range at a predefined address without spilling.
    /// The range is registered only if it does not overlap with any existing live range;
    /// otherwise allocation fails.
    /// @param newRange The live range to allocate (must already have an address assigned)
    /// @return true if allocation successful, false if the range overlaps with an existing live range
    bool allocDefinedRange(const LiveRange& newRange) {
        AllocationGuard guard(_partitioner);

        const auto newRangeAddr = _handler.getAddress(newRange);
        const auto newRangeSize = _handler.getSize(newRange);

        // verify no overlap
        for (const auto& prevRange : _liveRanges) {
            const auto prevRangeAddr = _handler.getAddress(prevRange);
            const auto prevRangeSize = _handler.getSize(prevRange);

            if (Partitioner::intersects(newRangeAddr, newRangeSize, prevRangeAddr, prevRangeSize)) {
                return false;
            }
        }

        _partitioner.allocFixed(newRangeAddr, newRangeSize);

        _liveRanges.push_back(newRange);

        guard.commit();
        return true;
    }

    /// Allocates live ranges with additional reserved space.
    /// First reserves the specified space, allocates ranges, then frees the reserved space.
    /// @param newLiveRanges Collection of live ranges to allocate
    /// @param reservedSpaceSize Size of the reserved space
    /// @param baseAlignment Alignment requirement for the reserved space
    /// @param allowSpills If true, allows spilling of existing ranges
    /// @param dir Allocation direction (Up or Down)
    /// @return Address of reserved space if successful, InvalidAddress otherwise
    template <class LiveRanges>
    vpux::AddressType allocWithReservedSpace(const LiveRanges& newLiveRanges, vpux::AddressType reservedSpaceSize,
                                             vpux::AddressType baseAlignment, bool allowSpills = true,
                                             Direction dir = Direction::Up) {
        auto allocAddr = _partitioner.alloc(reservedSpaceSize, baseAlignment, dir);
        if (allocAddr == InvalidAddress) {
            return InvalidAddress;
        }

        bool allocSuccess = alloc(newLiveRanges, allowSpills, dir);
        _partitioner.free(allocAddr, reservedSpaceSize);

        return allocSuccess ? allocAddr : InvalidAddress;
    }

    /// Allocates live ranges while excluding certain memory regions.
    /// Temporarily marks excluded regions as allocated, then allocates new ranges in remaining space.
    /// @param excludedRanges Vector of (address, size) pairs to temporarily exclude
    /// @param newLiveRanges Collection of live ranges to allocate
    /// @param allowSpills If true, allows spilling of existing ranges
    /// @param dir Allocation direction (Up or Down)
    /// @return true if all ranges allocated successfully, false otherwise and no changes are made
    template <class LiveRanges>
    bool allocWithExcludedRegion(ReservedAddressAndSizeVector excludedRanges, const LiveRanges& newLiveRanges,
                                 bool allowSpills = true, Direction dir = Direction::Up) {
        SmallVector<std::pair<vpux::AddressType, vpux::AddressType>> tempAlloc;

        for (const auto& [rangeAddr, rangeSize] : excludedRanges) {
            _partitioner.allocFixed(rangeAddr, rangeSize);
            tempAlloc.push_back({rangeAddr, rangeSize});
        }

        bool allocSuccess = alloc(newLiveRanges, allowSpills, dir);

        for (const auto& [address, size] : tempAlloc) {
            _partitioner.free(address, size);
        }

        return allocSuccess;
    }

    /// Checks if the given ranges can be allocated without spilling.
    /// Note: This method temporarily modifies the partitioner state (allocate + free)
    /// to test allocation feasibility. The state is restored after the check.
    /// @param newLiveRanges The ranges to test for allocation
    /// @param dir Allocation direction (Up or Down)
    /// @return true if all ranges can be allocated, false otherwise
    template <class LiveRanges>
    bool canAlloc(const LiveRanges& newLiveRanges, Direction dir = Direction::Up) {
        TempAllocationTester tester(_partitioner);

        for (const auto& newRange : newLiveRanges) {
            const auto size = _handler.getSize(newRange);
            const auto alignment = _handler.getAlignment(newRange);

            if (tester.alloc(size, alignment, dir) == InvalidAddress) {
                return false;
            }
        }

        return true;
    }

    /// Checks if ranges can be allocated with reserved space.
    /// Tests allocation feasibility without actually allocating (state is restored after check).
    /// @param newLiveRanges Collection of live ranges to test for allocation
    /// @param reservedSpaceSize Size of the reserved space that will be allocated first
    /// @param baseAlignment Alignment requirement for the reserved space
    /// @param dir Allocation direction: Direction::Up or Direction::Down
    /// @return true if allocation is possible, false otherwise
    template <class LiveRanges>
    bool canAllocWithReservedSpace(const LiveRanges& newLiveRanges, vpux::AddressType reservedSpaceSize,
                                   vpux::AddressType baseAlignment, Direction dir = Direction::Up) {
        TempAllocationTester tester(_partitioner);

        if (tester.alloc(reservedSpaceSize, baseAlignment, dir) == InvalidAddress) {
            return false;
        }

        for (const auto& newRange : newLiveRanges) {
            const auto size = _handler.getSize(newRange);
            const auto alignment = _handler.getAlignment(newRange);

            if (tester.alloc(size, alignment, dir) == InvalidAddress) {
                return false;
            }
        }

        return true;
    }

    /// Checks if ranges can be allocated while excluding certain regions.
    /// Tests allocation feasibility with temporarily excluded regions (state is restored after check).
    /// @param excludedRanges Vector of (address, size) pairs to temporarily exclude from allocation
    /// @param newLiveRanges Collection of live ranges to test for allocation
    /// @param dir Allocation direction: Direction::Up or Direction::Down
    /// @return true if allocation is possible, false otherwise
    template <class LiveRanges>
    bool canAllocWithExcludedRegion(ReservedAddressAndSizeVector excludedRanges, const LiveRanges& newLiveRanges,
                                    Direction dir = Direction::Up) {
        TempAllocationTester tester(_partitioner);

        for (const auto& [addr, size] : excludedRanges) {
            tester.allocFixed(addr, size);
        }

        for (const auto& newRange : newLiveRanges) {
            const auto size = _handler.getSize(newRange);
            const auto alignment = _handler.getAlignment(newRange);

            if (tester.alloc(size, alignment, dir) == InvalidAddress) {
                return false;
            }
        }

        return true;
    }

    // Overloads for initializer_list - necessary because template deduction
    // cannot deduce types from braced-init-lists like {&r1, &r2}
    bool alloc(std::initializer_list<LiveRange> newLiveRanges, bool allowSpills = true, Direction dir = Direction::Up) {
        return alloc<std::initializer_list<LiveRange>>(newLiveRanges, allowSpills, dir);
    }

    vpux::AddressType allocWithReservedSpace(std::initializer_list<LiveRange> newLiveRanges,
                                             vpux::AddressType reservedSpaceSize, vpux::AddressType baseAlignment,
                                             bool allowSpills = true, Direction dir = Direction::Up) {
        return allocWithReservedSpace<std::initializer_list<LiveRange>>(newLiveRanges, reservedSpaceSize, baseAlignment,
                                                                        allowSpills, dir);
    }

    bool allocWithExcludedRegion(ReservedAddressAndSizeVector excludedRanges,
                                 std::initializer_list<LiveRange> newLiveRanges, bool allowSpills = true,
                                 Direction dir = Direction::Up) {
        return allocWithExcludedRegion<std::initializer_list<LiveRange>>(excludedRanges, newLiveRanges, allowSpills,
                                                                         dir);
    }

    bool canAlloc(std::initializer_list<LiveRange> newLiveRanges, Direction dir = Direction::Up) {
        return canAlloc<std::initializer_list<LiveRange>>(newLiveRanges, dir);
    }

    bool canAllocWithReservedSpace(std::initializer_list<LiveRange> newLiveRanges, vpux::AddressType reservedSpaceSize,
                                   vpux::AddressType baseAlignment, Direction dir = Direction::Up) {
        return canAllocWithReservedSpace<std::initializer_list<LiveRange>>(newLiveRanges, reservedSpaceSize,
                                                                           baseAlignment, dir);
    }

    bool canAllocWithExcludedRegion(ReservedAddressAndSizeVector excludedRanges,
                                    std::initializer_list<LiveRange> newLiveRanges, Direction dir = Direction::Up) {
        return canAllocWithExcludedRegion<std::initializer_list<LiveRange>>(excludedRanges, newLiveRanges, dir);
    }

public:
    /// Returns the current set of live ranges
    const auto& liveRanges() const {
        return _liveRanges;
    }

    /// Returns const reference to the handler
    const auto& handler() const {
        return _handler;
    }

    /// Returns mutable reference to the handler
    auto& handler() {
        return _handler;
    }

    /// Returns the total size of the memory region
    AddressType totalSize() const {
        return _partitioner.totalSize();
    }

    /// Returns the total free size across all gaps
    AddressType totalFreeSize() const {
        return _partitioner.totalFreeSize();
    }

    /// Returns the maximum allocated size.
    /// The handler tracks the high-water mark of the highest end-address (addr + size) across
    /// buffer allocations reported via handler.allocated(). However, reserved regions set up in
    /// the constructor or via allocWithReservedSpace/allocWithExcludedRegion are allocated only
    /// in the partitioner and are not reported to the handler. Taking the max of current
    /// partitioner usage and the handler high-water mark ensures reserved space is accounted for.
    AddressType maxAllocatedSize() const {
        auto totalUsedSize = _partitioner.totalSize() - _partitioner.totalFreeSize();
        auto handlerMaxAllocatedSize = static_cast<uint64_t>(_handler.maxAllocatedSize().count());
        return std::max(totalUsedSize, handlerMaxAllocatedSize);
    }

    /// Returns the size of the largest free gap
    AddressType maxFreeSize() const {
        return _partitioner.maxFreeSize();
    }

    /// Returns the current memory gaps
    const auto& gaps() const {
        return _partitioner.gaps();
    }

private:
    /// RAII helper for temporary allocation testing with automatic cleanup
    class TempAllocationTester final {
    public:
        explicit TempAllocationTester(Partitioner& partitioner)
                : _partitioner(partitioner), _gapCountBefore(partitioner.gaps().size()) {
        }

        ~TempAllocationTester() {
            for (const auto& [addr, size] : _tempAllocations) {
                _partitioner.free(addr, size);
            }
            assert(_partitioner.gaps().size() == _gapCountBefore &&
                   "Error: gap count changed after temporary allocation test");
        }

        // Non-copyable, non-movable
        TempAllocationTester(const TempAllocationTester&) = delete;
        TempAllocationTester& operator=(const TempAllocationTester&) = delete;
        TempAllocationTester(TempAllocationTester&&) = delete;
        TempAllocationTester& operator=(TempAllocationTester&&) = delete;

        /// Records a temporary allocation for later cleanup.
        /// @param addr Address of the allocation
        /// @param size Size of the allocation
        void recordAllocation(AddressType addr, AddressType size) {
            if (addr != InvalidAddress) {
                _tempAllocations.push_back({addr, size});
            }
        }

        /// Allocates memory temporarily (freed automatically in destructor).
        /// @param size Size of the allocation
        /// @param alignment Alignment requirement
        /// @param dir Allocation direction (Up or Down)
        /// @return Allocated address, or InvalidAddress if allocation fails
        AddressType alloc(AddressType size, AddressType alignment, Direction dir) {
            auto addr = _partitioner.alloc(size, alignment, dir);
            recordAllocation(addr, size);
            return addr;
        }

        /// Allocates at a fixed address temporarily (freed automatically in destructor).
        /// @param addr Address to allocate at
        /// @param size Size of the allocation
        void allocFixed(AddressType addr, AddressType size) {
            _partitioner.allocFixed(addr, size);
            recordAllocation(addr, size);
        }

    private:
        Partitioner& _partitioner;
        SmallVector<std::pair<AddressType, AddressType>> _tempAllocations;
        size_t _gapCountBefore;
    };

    /// RAII guard for exception-safe allocation with automatic rollback
    class AllocationGuard final {
    public:
        explicit AllocationGuard(Partitioner& partitioner): _partitioner(partitioner), _committed(false) {
        }

        // Non-copyable
        AllocationGuard(const AllocationGuard&) = delete;
        AllocationGuard& operator=(const AllocationGuard&) = delete;

        // Movable
        AllocationGuard(AllocationGuard&& other) noexcept
                : _partitioner(other._partitioner),
                  _allocations(std::move(other._allocations)),
                  _committed(other._committed) {
            other._committed = true;
        }

        AllocationGuard& operator=(AllocationGuard&& other) noexcept {
            if (this != &other) {
                if (!_committed) {
                    rollback();
                }
                _allocations = std::move(other._allocations);
                _committed = other._committed;
                other._committed = true;
            }
            return *this;
        }

        ~AllocationGuard() {
            if (!_committed) {
                rollback();
            }
        }

        /// Records an allocation for commit or rollback.
        /// @param addr Address of the allocation
        /// @param size Size of the allocation
        void recordAllocation(AddressType addr, AddressType size) {
            if (addr != InvalidAddress) {
                _allocations.push_back({addr, size});
            }
        }

        /// Commits all recorded allocations, preventing rollback on destruction.
        void commit() {
            _committed = true;
        }

        /// Checks if allocations have been committed.
        /// @return true if committed, false if will rollback on destruction
        bool isCommitted() const {
            return _committed;
        }

    private:
        /// Rolls back all recorded allocations by freeing them.
        void rollback() {
            for (const auto& [addr, size] : _allocations) {
                _partitioner.free(addr, size);
            }
            _allocations.clear();
        }

        Partitioner& _partitioner;
        SmallVector<std::pair<AddressType, AddressType>> _allocations;
        bool _committed;
    };

    /// Allocates a live range at a fixed address with optional spilling.
    /// @param newRange The live range to allocate (must be fixed allocation)
    /// @param allowSpills If true, allows spilling of conflicting ranges
    /// @param guard AllocationGuard to track this allocation
    /// @return true if allocation successful, false otherwise
    bool allocFixedRange(const LiveRange& newRange, bool allowSpills, AllocationGuard& guard) {
        assert(_handler.isFixedAlloc(newRange));

        const auto newRangeAddr = _handler.getAddress(newRange);
        const auto newRangeSize = _handler.getSize(newRange);
        assert(newRangeAddr != InvalidAddress);

        const auto canAlloc = [this, newRangeAddr, newRangeSize]() {
            for (const auto& prevRange : _liveRanges) {
                const auto prevRangeAddr = _handler.getAddress(prevRange);
                const auto prevRangeSize = _handler.getSize(prevRange);

                if (Partitioner::intersects(newRangeAddr, newRangeSize, prevRangeAddr, prevRangeSize)) {
                    return false;
                }
            }

            return true;
        };

        LiveRangeVector spillCandidates;

        for (;;) {
            if (canAlloc()) {
                break;
            }

            if (!allowSpills) {
                return false;
            }

            const auto spillCandidate = getSpillCandidate();

            if (!spillCandidate.has_value()) {
                return false;
            }

            spillCandidates.push_back(spillCandidate.value());
        }

        if (!processSpillCandidates(spillCandidates, newRangeAddr, newRangeSize)) {
            return false;
        }

        _partitioner.allocFixed(newRangeAddr, newRangeSize);
        guard.recordAllocation(newRangeAddr, newRangeSize);

        return true;
    }

    /// Allocates a live range at a dynamic address with optional spilling.
    /// @param newRange The live range to allocate (must not be fixed allocation)
    /// @param allowSpills If true, allows spilling of conflicting ranges
    /// @param dir Allocation direction (Up or Down)
    /// @param guard AllocationGuard to track this allocation
    /// @param outAddr [out] The allocated address if successful
    /// @return true if allocation successful, false otherwise
    bool allocDynamicRange(const LiveRange& newRange, bool allowSpills, Direction dir, AllocationGuard& guard,
                           AddressType& outAddr) {
        assert(!_handler.isFixedAlloc(newRange));

        const auto newRangeSize = _handler.getSize(newRange);
        const auto newRangeAlignment = _handler.getAlignment(newRange);

        auto newRangeAddr = InvalidAddress;

        LiveRangeVector spillCandidates;

        for (;;) {
            newRangeAddr = _partitioner.alloc(newRangeSize, newRangeAlignment, dir);

            if (newRangeAddr != InvalidAddress) {
                break;
            }

            if (!allowSpills) {
                return false;
            }

            const auto spillCandidate = getSpillCandidate();

            if (!spillCandidate.has_value()) {
                return false;
            }

            spillCandidates.push_back(spillCandidate.value());
        }

        if (!processSpillCandidates(spillCandidates, newRangeAddr, newRangeSize)) {
            return false;
        }

        guard.recordAllocation(newRangeAddr, newRangeSize);
        outAddr = newRangeAddr;

        return true;
    }

    /// Frees a live range at the specified index using swap-and-pop.
    /// This is an O(1) operation that swaps the element with the last one and removes it.
    /// @param idx Index of the live range to free (must be valid)
    void freeLiveRangeAt(size_t idx) {
        assert(idx < _liveRanges.size());
        const auto& range = _liveRanges[idx];

        const auto addr = _handler.getAddress(range);
        const auto size = _handler.getSize(range);

        _partitioner.free(addr, size);
        _handler.freed(range);

        _liveRanges[idx] = _liveRanges.back();
        _liveRanges.pop_back();
    }

    /// Find and remove the live range with lowest spill weight.
    /// @return Selected spill candidate, or std::nullopt if none available
    std::optional<LiveRange> getSpillCandidate() {
        const auto it = std::min_element(_liveRanges.begin(), _liveRanges.end(),
                                         [this](const LiveRange& r1, const LiveRange& r2) {
                                             return _handler.getSpillWeight(r1) < _handler.getSpillWeight(r2);
                                         });

        if (it == _liveRanges.end()) {
            return std::nullopt;
        }

        const auto spillCandidate = *it;

        const auto spilledAddr = _handler.getAddress(spillCandidate);
        const auto spilledSize = _handler.getSize(spillCandidate);

        _partitioner.free(spilledAddr, spilledSize);

        *it = _liveRanges.back();
        _liveRanges.pop_back();

        return spillCandidate;
    }

    /// Process spill candidates: spill conflicting ranges, restore non-conflicting ones.
    /// @return true if all conflicting ranges successfully spilled, false otherwise
    bool processSpillCandidates(const LiveRangeVector& spillCandidates, AddressType newRangeAddr,
                                AddressType newRangeSize) {
        for (const auto& candidate : spillCandidates) {
            const auto candidateAddr = _handler.getAddress(candidate);
            const auto candidateSize = _handler.getSize(candidate);

            if (Partitioner::intersects(candidateAddr, candidateSize, newRangeAddr, newRangeSize)) {
                if (!spillConflictingRange(candidate)) {
                    return false;
                }
            } else {
                restoreNonConflictingRange(candidate, candidateAddr, candidateSize);
            }
        }

        return true;
    }

private:
    /// Spills a conflicting live range by calling the handler's spill method.
    /// @param range The live range to spill
    /// @return true if spilling successful, false otherwise
    bool spillConflictingRange(const LiveRange& range) {
        if (!_handler.spilled(range)) {
            return false;
        }
        _handler.freed(range);
        return true;
    }

    /// Restores a non-conflicting live range back to the allocated list.
    /// @param range The live range to restore
    /// @param addr The address to allocate at
    /// @param size The size to allocate
    void restoreNonConflictingRange(const LiveRange& range, AddressType addr, AddressType size) {
        _partitioner.allocFixed(addr, size);
        _handler.allocated(range, addr);
        _liveRanges.push_back(range);
    }

private:
    Partitioner _partitioner;
    LiveRangeVector _liveRanges;
    Handler _handler;
};

}  // namespace vpux
