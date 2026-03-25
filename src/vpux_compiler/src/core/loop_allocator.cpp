//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/loop_allocator.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <algorithm>

using namespace vpux;

LoopAllocator::LoopAllocator(ComputeRegionVec& computeRegionVec, vpux::AddressType memorySize, Logger log,
                             llvm::StringLiteral logName)
        : _computeRegions(computeRegionVec), _memorySize(memorySize), _log(log) {
    _log.setName(logName);
}

vpux::AddressType LoopAllocator::getAlignedBufferSize(mlir::Value buf) {
    return alignValUp(static_cast<vpux::AddressType>(vpux::getTotalSize(buf).count()), vpux::getAlignment(buf));
}

vpux::AddressType LoopAllocator::getRawBufferSize(mlir::Value buf) {
    return static_cast<vpux::AddressType>(vpux::getTotalSize(buf).count());
}

vpux::AddressType LoopAllocator::getBufferAlignment(mlir::Value buf) {
    return vpux::getAlignment(buf);
}

void LoopAllocator::analyzeLoopBuffers(const SchedulingLoop& schedulingLoop) {
    _profile = LoopBufferProfile();
    SmallVector<vpux::AddressType> maxBufferSizes;
    mlir::DenseMap<mlir::Value, size_t> bufferUseCount;
    vpux::AddressType largestLoopSize = 0;
    // Find the largest schedulingLoop
    for (size_t loopIdx = 0; loopIdx < schedulingLoop.loopBodies.size(); ++loopIdx) {
        const auto& allocInfoVec = schedulingLoop.loopBodies[loopIdx];
        mlir::DenseSet<mlir::Value> handledBuffers;
        vpux::AddressType loopSize = 0;
        for (const auto& allocInfo : allocInfoVec) {
            if (allocInfo.allocationType != AllocationType::COMPUTE) {
                continue;
            }
            for (const auto& buf : allocInfo.inBuffers) {
                if (handledBuffers.count(buf) > 0) {
                    continue;
                }
                bufferUseCount[buf]++;
                handledBuffers.insert(buf);
                loopSize += getAlignedBufferSize(buf);
            }
            for (const auto& buf : allocInfo.outBuffers) {
                if (handledBuffers.count(buf) > 0) {
                    continue;
                }
                bufferUseCount[buf]++;
                handledBuffers.insert(buf);
                loopSize += getAlignedBufferSize(buf);
            }
        }

        if (largestLoopSize < loopSize) {
            largestLoopSize = loopSize;
            _profile.largestLoopIdx = loopIdx;
        }
    }

    // Build maxBufferSizes from the largest schedulingLoop iteration
    mlir::DenseSet<mlir::Value> processedBuffers;
    for (const auto& allocInfo : schedulingLoop.loopBodies[_profile.largestLoopIdx]) {
        if (allocInfo.allocationType != AllocationType::COMPUTE) {
            continue;
        }
        for (const auto& buf : allocInfo.inBuffers) {
            if (processedBuffers.count(buf) > 0) {
                continue;
            }
            processedBuffers.insert(buf);
            maxBufferSizes.push_back(getAlignedBufferSize(buf));
        }
        for (const auto& buf : allocInfo.outBuffers) {
            if (processedBuffers.count(buf) > 0) {
                continue;
            }
            processedBuffers.insert(buf);
            maxBufferSizes.push_back(getAlignedBufferSize(buf));
        }
    }

    // Validate buffer order and sizes for other iterations
    for (size_t idx = 0; idx < schedulingLoop.loopBodies.size(); ++idx) {
        if (idx == _profile.largestLoopIdx) {
            continue;
        }
        mlir::DenseSet<mlir::Value> processedLoopBuffers;
        size_t loopBufferIdx = 0;
        for (const auto& allocInfo : schedulingLoop.loopBodies[idx]) {
            if (allocInfo.allocationType != AllocationType::COMPUTE) {
                continue;
            }
            for (const auto& buf : allocInfo.inBuffers) {
                if (processedLoopBuffers.count(buf) > 0) {
                    continue;
                }
                processedLoopBuffers.insert(buf);
                auto bufSize = getAlignedBufferSize(buf);
                VPUX_THROW_WHEN(loopBufferIdx >= maxBufferSizes.size() || maxBufferSizes[loopBufferIdx] < bufSize,
                                "Inconsistent buffer count OR sizes in loop tiling region");
                ++loopBufferIdx;
            }
            for (const auto& buf : allocInfo.outBuffers) {
                if (processedLoopBuffers.count(buf) > 0) {
                    continue;
                }
                processedLoopBuffers.insert(buf);
                auto bufSize = getAlignedBufferSize(buf);
                VPUX_THROW_WHEN(loopBufferIdx >= maxBufferSizes.size() || maxBufferSizes[loopBufferIdx] < bufSize,
                                "Inconsistent buffer count OR sizes in loop tiling region");
                ++loopBufferIdx;
            }
        }
    }

    // Verify buffer usage 1(local) or All(shared) and collect shared buffers
    for (const auto& [buf, count] : bufferUseCount) {
        VPUX_THROW_WHEN(count != 1 && count != schedulingLoop.loopBodies.size(), "Buffer has invalid use count {0}",
                        count);
        if (count == schedulingLoop.loopBodies.size()) {
            _profile.sharedBuffers.insert(buf);
        }
    }
}

void LoopAllocator::reserveSharedBuffers() {
    auto nextFreeOffsetFromTop = _memorySize;
    for (const auto& buf : _profile.sharedBuffers) {
        nextFreeOffsetFromTop -= getRawBufferSize(buf);
        nextFreeOffsetFromTop = alignValDown(nextFreeOffsetFromTop, getBufferAlignment(buf));
        _log.nest(2).trace("shared buffer size {0} address {1}", getAlignedBufferSize(buf), nextFreeOffsetFromTop);
    }

    _profile.sharedRegionSize = _memorySize - nextFreeOffsetFromTop;
    VPUX_THROW_WHEN(_profile.sharedRegionSize > _memorySize,
                    "Not enough memory for shared buffers in loop tiling region");

    _log.nest().trace("sharedRegionSize {0}", _profile.sharedRegionSize);
}

bool LoopAllocator::tryAllocateBuffer(PingPongBuffer& bufferSet, mlir::Value buf, bool isAllocateFirstOp,
                                      llvm::StringRef bufferRole) {
    // Skip shared buffers (already allocated top-down)
    if (_profile.sharedBuffers.count(buf) > 0) {
        return true;
    }

    // Skip if already allocated in this set
    if (bufferSet.addressMap.count(buf) > 0) {
        _log.nest(3).trace("reused buffer address {0}", bufferSet.addressMap[buf]);
        return true;
    }

    if (bufferSet.reachedMemoryLimit) {
        return false;
    }

    if (bufferSet.addressMap.empty()) {
        bufferSet.baseAlignment = getBufferAlignment(buf);
    }

    const auto alignedOffset = alignValUp(bufferSet.nextFreeOffset, getBufferAlignment(buf));
    const auto nextOffset = alignedOffset + getRawBufferSize(buf);
    if (_profile.sharedRegionSize + nextOffset > _memorySize) {
        VPUX_THROW_WHEN(isAllocateFirstOp, "OOM on first op of 1st ping-pong buffer set, cannot proceed");
        _log.nest().trace("cannot prefetch more {0} + {1} > {2}", _profile.sharedRegionSize, nextOffset, _memorySize);
        bufferSet.reachedMemoryLimit = true;
        return false;
    }

    bufferSet.addressMap[buf] = alignedOffset;
    bufferSet.addressSequence.push_back(alignedOffset);
    bufferSet.nextFreeOffset = nextOffset;
    bufferSet.prefetchBufferCount = bufferSet.addressSequence.size();
    bufferSet.reserveLocalRegionSize = bufferSet.nextFreeOffset;
    _log.nest(3).trace("allocated {0} buffer size {1} address {2}", bufferRole, getAlignedBufferSize(buf),
                       bufferSet.addressMap[buf]);
    return true;
}

// Allocates a single Set of Ping-Pong buffers (either the 1st/Even set or 2nd/Odd set).
//
// Logic:
// 1. Scans the Loop Body sequentially.
// 2. For each buffer required by operations:
//    - Skips if it's a Shared Buffer (already reserved at the top).
//    - Allocates it bottom-up from 'initialOffset'.
// 3. For 2nd Set: If memory limit is reached during an op, we break out of the loop.
//    The buffers allocated in that op (before OOM) will be overwritten with 1st Set addresses.
//
// Parameters:
// - initialOffset: Starting address for this set (0 for 1st Set; End-of-1st-Set for 2nd Set).
// - isAllocateFirstOp:
//     - True (1st Set): Must succeed for at least one Op, or we throw (Out of Memory).
//     - False (2nd Set): Can fail gracefully, breaks on OOM and tracks prefetchBufferCount.
//
// Returns:
// - PingPongBuffer: The allocated buffer set state.
// - size_t (prefetchOpCount): The number of operations that completed successfully.
std::pair<PingPongBuffer, size_t> LoopAllocator::allocatePingPongBuffer(const LoopBody& loopBody,
                                                                        vpux::AddressType initialOffset,
                                                                        bool isAllocateFirstOp) {
    PingPongBuffer bufferSet;
    bufferSet.nextFreeOffset = initialOffset;
    bufferSet.reserveLocalRegionSize = initialOffset;

    size_t prefetchOpCount = 0;

    for (const auto& allocInfo : loopBody) {
        _log.nest(2).trace("scheduling {0} loop, op : {1}", isAllocateFirstOp ? "even" : "odd", allocInfo.opIdx);
        for (const auto& buf : allocInfo.inBuffers) {
            if (!tryAllocateBuffer(bufferSet, buf, isAllocateFirstOp, "in")) {
                break;
            }
        }
        if (bufferSet.reachedMemoryLimit) {
            break;
        }
        for (const auto& buf : allocInfo.outBuffers) {
            if (!tryAllocateBuffer(bufferSet, buf, isAllocateFirstOp, "out")) {
                break;
            }
        }
        if (bufferSet.reachedMemoryLimit) {
            break;
        }

        ++prefetchOpCount;
    }

    return {bufferSet, prefetchOpCount};
}

void LoopAllocator::validateDependencies(const ComputeRegion& computeRegion) {
    for (auto dep : computeRegion.dependencies) {
        auto it = _externalDepsLookUp.find(dep);
        if (it == _externalDepsLookUp.end()) {
            continue;
        }
        auto allocInfo = it->second;
        if (allocInfo.allocationType != AllocationType::DATA_IN) {
            continue;
        }
        for (const auto& buf : allocInfo.outBuffers) {
            VPUX_THROW_WHEN(_profile.sharedBuffers.count(buf) == 0, "Dependency buffer not in shared buffers");
        }
    }
}

void LoopAllocator::allocateLoopTilingRegion(ComputeRegion& computeRegion) {
    _log.trace("Compute region: {0}", computeRegion);

    const auto& loopBodies = computeRegion.schedulingLoop->loopBodies;
    const auto computeCount = std::count_if(loopBodies[0].begin(), loopBodies[0].end(), [](const auto& allocInfo) {
        return allocInfo.allocationType == AllocationType::COMPUTE;
    });
    VPUX_THROW_WHEN(computeCount != 1, "No support for multiple compute ops in loop tiling region -> should be VF");

    // Analyze buffers and build profile
    analyzeLoopBuffers(*computeRegion.schedulingLoop);
    validateDependencies(computeRegion);

    // Reserve space for shared buffers (from top)
    reserveSharedBuffers();

    // Allocate ping-pong buffer sets

    // 1. Allocate 1st Set (Even Iterations)
    //    - Starts at offset 0 (relative to local memory start).
    //    - Must succeed (isAllocateFirstOp=true). If OOM here, we can't run even a single iteration.
    auto [firstSet, _] = allocatePingPongBuffer(loopBodies[_profile.largestLoopIdx], /*initialOffset=*/0,
                                                /*isAllocateFirstOp*/ true);

    // 2. Allocate 2nd Set (Odd Iterations) for Pipelining
    //    - Starts immediately after the 1st Set.
    //    - Can fail partially or completely (isAllocateFirstOp=false).
    //    - If OOM, we get 'prefetchOpCount' indicating how many ops were successfully doubled-buffered.
    //    - Remaining ops will reuse 1st Set addresses (Partial Pipelining).
    auto [secondSet, prefetchOpCount] =
            allocatePingPongBuffer(loopBodies[_profile.largestLoopIdx], firstSet.nextFreeOffset,
                                   /*isAllocateFirstOp*/ false);

    _log.nest().trace("prefetchOpCount {0}, reserveLocalRegionSize {1}", prefetchOpCount,
                      secondSet.reserveLocalRegionSize);

    // Second set may have fewer buffers if memory ran out during prefetch.
    // This is OK, we'll reuse addresses from first set for the remaining buffers.
    if (secondSet.prefetchBufferCount < firstSet.addressSequence.size()) {
        _log.nest().trace("No pipelining possible for compute region {0}", computeRegion);
    }

    // Copy remaining addresses from first set (for buffers that couldn't be prefetched)
    size_t prefetchBufferCount = secondSet.prefetchBufferCount;
    for (; prefetchBufferCount < secondSet.addressSequence.size(); prefetchBufferCount++) {
        secondSet.addressSequence[prefetchBufferCount] = firstSet.addressSequence[prefetchBufferCount];
    }
    // Append addresses that weren't even allocated in 2nd set
    for (; prefetchBufferCount < firstSet.addressSequence.size(); prefetchBufferCount++) {
        secondSet.addressSequence.push_back(firstSet.addressSequence[prefetchBufferCount]);
    }

    VPUX_THROW_WHEN(firstSet.addressSequence.size() != secondSet.addressSequence.size(),
                    "Inconsistent prefetch address vector size");
    for (size_t i = 0; i < firstSet.addressSequence.size(); i++) {
        _log.nest().trace("pingPongAddr[{0}] = {1}, {2}", i, firstSet.addressSequence[i], secondSet.addressSequence[i]);
    }

    computeRegion.size = secondSet.reserveLocalRegionSize;
    computeRegion.bufferAddressVec =
            std::make_pair(std::move(firstSet.addressSequence), std::move(secondSet.addressSequence));
    computeRegion.sharedExternalBuffers = std::move(_profile.sharedBuffers);
    computeRegion.baseAlignment = firstSet.baseAlignment;
    computeRegion.prefetchOpCount = prefetchOpCount;
}

void LoopAllocator::buildExternalDepsLookUp() {
    for (const auto& computeRegion : _computeRegions) {
        if (computeRegion.schedulingLoop == nullptr || computeRegion.schedulingLoop->type != LoopType::None) {
            continue;
        }
        for (const auto& loopBody : computeRegion.schedulingLoop->loopBodies) {
            for (const auto& allocation : loopBody) {
                _externalDepsLookUp[allocation.opIdx] = allocation;
            }
        }
    }
}

void LoopAllocator::allocateLoopTilingRegions() {
    buildExternalDepsLookUp();
    for (auto& computeRegion : _computeRegions) {
        if (computeRegion.schedulingLoop == nullptr || computeRegion.schedulingLoop->type != LoopType::Tiling) {
            continue;
        }
        allocateLoopTilingRegion(computeRegion);
    }
}
