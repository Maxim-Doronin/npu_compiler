//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <unordered_map>

namespace vpux {

/// Loop Allocator handles CMX memory allocation for loop tiling regions.
///
/// This allocator manages memory layout to enable efficient pipelining for tiled operations.
///
/// 1. Shared Buffers:
///    Buffers used across all iterations (e.g., weights, constants) are allocated from the
///    top of CMX memory downwards.
///
///    NOTE: We do NOT assign final addresses to them here. Instead, we only reserve their
///    total size to ensure sufficient space for Local Buffers. The actual allocation happens
///    in the global Feasible Allocation pass, allowing Shared Buffers to participate in
///    global prefetching and pipelining alongside non-tiled operations. We avoid binding
///    Shared Buffers with Local Buffers to prevent creating large monolithic memory blocks
///    that would cause fragmentation and hinder scheduling flexibility.
///
/// 2. Local Buffers (Ping-Pong):
///    Buffers specific to each iteration (input/output tiles) are allocated from address 0
///    upwards. We attempt to allocate two sets (ping-pong) to enable pipelining:
///    - 1st Set: Allocated at the bottom (Address 0+). Used by even tiles.
///    - 2nd Set: Allocated immediately after 1st set. Used by odd tiles.
///
///    If memory is insufficient for a full 2nd set, we prioritize prefetching what fits
///    and reuse 1st set addresses for the rest, leading to partial pipelining.
///
///    The contiguous memory region allocated for Local Buffers (containing the ping-pong sets)
///    is treated as a single cohesive group during the Feasible Allocation pass.
///    This guarantees that the tiled operations are "perfectly scheduled" as a unit,
///    preserving the internal ping-pong structure required for pipelining without being
///    broken apart by global scheduler decisions.
///
/// Memory Layout Scenarios:
///
/// Case 1: No prefetching
/// CMX buffer only enough for the first iteration.
///
/// ┌───────────────────────────┐ ← memorySize
/// │ Shared Buffers            │
/// ├───────────────────────────┤ ← firstSet.nextFreeOffset
/// │ Buffer 4 (1st set)        │ ← reuse 2nd set
/// │ Buffer 3 (1st set)        │ ← reuse 2nd set
/// │ Buffer 2 (1st set)        │ ← reuse 2nd set
/// │ Buffer 1 (1st set)        │ ← reuse 2nd set
/// └───────────────────────────┘ ← 0
///
/// All tiles will reuse the same buffer.
///
/// Case 2: Prefetching but no pipelining (Partial Pipeline)
///
/// ┌───────────────────────────┐ ← memorySize
/// │ Shared Buffers            │
/// ├───────────────────────────┤
/// │ Buffer 2 (2nd set)        │ ← prefetch
/// │ Buffer 1 (2nd set)        │ ← prefetch
/// ├───────────────────────────┤ ← firstSet.nextFreeOffset
/// │ Buffer 4 (1st set)        │ ← reuse 2nd set
/// │ Buffer 3 (1st set)        │ ← reuse 2nd set
/// │ Buffer 2 (1st set)        │
/// │ Buffer 1 (1st set)        │
/// └───────────────────────────┘ ← 0
///
/// All even (0, 2, 4...) tiles use 1st set buffer (1, 2, 3, 4).
/// All odd (1, 3, 5...) tiles use 2nd set buffer (1, 2) and 1st set buffer (3, 4).
///
/// Case 3: Full pipelining
///
/// ┌───────────────────────────┐ ← memorySize
/// │ Shared Buffers            │
/// ├───────────────────────────┤
/// │ Buffer 4 (2nd set)        │ ← prefetch
/// │ Buffer 3 (2nd set)        │ ← prefetch
/// │ Buffer 2 (2nd set)        │ ← prefetch
/// │ Buffer 1 (2nd set)        │ ← prefetch
/// ├───────────────────────────┤ ← firstSet.nextFreeOffset
/// │ Buffer 4 (1st set)        │
/// │ Buffer 3 (1st set)        │
/// │ Buffer 2 (1st set)        │
/// │ Buffer 1 (1st set)        │
/// └───────────────────────────┘ ← 0
///
/// All even (0, 2, 4...) tiles use 1st set buffer (1, 2, 3, 4).
/// All odd (1, 3, 5...) tiles use 2nd set buffer (1, 2, 3, 4).
///
/// NOTE: Buffer 1, 2, 3, 4 represents the combination of this tile's input and output buffers.

/// @brief Profile of buffer usage across loop iterations.
/// Used to determine maximum size requirements and identify shared buffers.
struct LoopBufferProfile {
    /// Shared buffers
    ValueOrderedSet sharedBuffers;

    /// Total size of shared buffer region
    vpux::AddressType sharedRegionSize = 0;

    /// Index of the loop iteration with the largest total buffer size
    /// Used to determine which iteration's buffer order to use for allocation
    size_t largestLoopIdx = 0;
};

/// @brief State tracker for a single ping-pong buffer set (local buffers).
/// Tracks memory allocation progress from bottom-up.
struct PingPongBuffer {
    /// Buffer Value to allocated address mapping
    mlir::DenseMap<mlir::Value, vpux::AddressType> addressMap;

    /// Addresses in allocation order
    SmallVector<vpux::AddressType> addressSequence;

    /// Buffers already processed in this set
    mlir::DenseSet<mlir::Value> processedBuffers;

    /// Next available offset for bottom-up allocation
    vpux::AddressType nextFreeOffset = 0;

    /// OOM flag for partial prefetch (2nd set only)
    bool reachedMemoryLimit = false;

    /// Base alignment
    vpux::AddressType baseAlignment = 64;

    /// Reserved local buffer region size for current set
    vpux::AddressType reserveLocalRegionSize = 0;

    /// Number of buffers successfully prefetched before OOM
    size_t prefetchBufferCount = 0;
};

class LoopAllocator final {
public:
    explicit LoopAllocator(ComputeRegionVec& computeRegionVec, vpux::AddressType memorySize, Logger log,
                           llvm::StringLiteral logName);

    void allocateLoopTilingRegions();

private:
    void allocateLoopTilingRegion(ComputeRegion& computeRegion);

    /// @brief Analyze buffer usage patterns across loop iterations
    void analyzeLoopBuffers(const SchedulingLoop& schedulingLoop);

    /// @brief Reserve space for shared buffers from top of memory
    void reserveSharedBuffers();

    /// @brief Allocate a single buffer in the ping-pong set
    bool tryAllocateBuffer(PingPongBuffer& bufferSet, mlir::Value buf, bool isAllocateFirstOp,
                           llvm::StringRef bufferRole);

    /// @brief Allocate one ping-pong buffer set
    /// @return Allocated buffer set and prefetchOpCount
    std::pair<PingPongBuffer, size_t> allocatePingPongBuffer(const LoopBody& loopBody, vpux::AddressType initialOffset,
                                                             bool isAllocateFirstOp);

    /// @brief Validate that all external dependencies use shared buffers
    void validateDependencies(const ComputeRegion& computeRegion);

    /// @brief Build lookup table for external dependencies.
    /// Tiled regions may depend on non-tiled operations (e.g., weights DMA_IN).
    /// This lookup bridges opIdx to OpAllocationInfo for dependency validation.
    void buildExternalDepsLookUp();

private:
    static vpux::AddressType getAlignedBufferSize(mlir::Value buf);
    static vpux::AddressType getRawBufferSize(mlir::Value buf);
    static vpux::AddressType getBufferAlignment(mlir::Value buf);

private:
    ComputeRegionVec& _computeRegions;
    vpux::AddressType _memorySize;
    Logger _log;
    /// Lookup table for external dependencies to resolve opIdx to OpAllocationInfo
    std::unordered_map<size_t, OpAllocationInfo> _externalDepsLookUp;
    LoopBufferProfile _profile;
};

}  // namespace vpux
