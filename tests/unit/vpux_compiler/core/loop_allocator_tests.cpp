//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/loop_allocator.hpp"
#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_LoopAllocator.*"

using namespace vpux;

class MLIR_LoopAllocator : public testing::Test {
protected:
    mlir::MLIRContext _ctx;
    mlir::OpBuilder _builder{&_ctx};
    std::unique_ptr<mlir::Block> _block;

    Logger _log = Logger::global();

    MLIR_LoopAllocator() {
        auto registry = vpux::createDialectRegistry();
        _ctx.appendDialectRegistry(registry);
        _ctx.loadAllAvailableDialects();

        _block = std::make_unique<mlir::Block>();
        _builder.setInsertionPointToStart(_block.get());
    }

    mlir::Value createBuffer(vpux::AddressType rawSize, vpux::AddressType rawAlign) {
        auto shape = SmallVector<int64_t>{static_cast<int64_t>(rawSize)};
        auto type = mlir::MemRefType::get(shape, _builder.getIntegerType(8));

        auto alloc = _builder.create<mlir::memref::AllocOp>(_builder.getUnknownLoc(), type);
        if (rawAlign != 0) {
            alloc.setAlignment(rawAlign);
        }
        return alloc;
    }

    OpAllocationInfo createComputeOp(size_t opIdx, const SmallVector<mlir::Value>& inBuffers,
                                     const SmallVector<mlir::Value>& outBuffers) {
        VPURT::TaskQueueType qt{config::ExecutorKind::DPU, 0};
        return OpAllocationInfo(opIdx, qt, inBuffers, outBuffers, AllocationType::COMPUTE);
    }
};

TEST_F(MLIR_LoopAllocator, BasicTilingUniformSizes) {
    // Tiling scenario:
    // - 2 iterations
    // - Each iteration has: 1 input buffer, 1 output buffer
    // - All buffers have the same size (1024 bytes, 64-byte aligned)

    ComputeRegionVec computeRegionVec;

    // Create compute region with tiling loop
    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    // Iteration 0
    LoopBody iteration0;
    {
        auto inBuf = createBuffer(/*rawSize=*/1024, /*rawAlign=*/64);
        auto outBuf = createBuffer(/*rawSize=*/1024, /*rawAlign=*/64);
        iteration0.push_back(createComputeOp(/*opIdx=*/0, {inBuf}, {outBuf}));
    }

    // Iteration 1
    LoopBody iteration1;
    {
        auto inBuf = createBuffer(/*rawSize=*/1024, /*rawAlign=*/64);
        auto outBuf = createBuffer(/*rawSize=*/1024, /*rawAlign=*/64);
        iteration1.push_back(createComputeOp(/*opIdx=*/1, {inBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1)};

    ComputeRegion region(std::move(schedulingLoop));
    computeRegionVec.push_back(std::move(region));

    // Create LoopAllocator and run allocation
    vpux::AddressType memorySize = 8192;  // 8KB
    LoopAllocator allocator(computeRegionVec, memorySize, _log, "test");
    allocator.allocateLoopTilingRegions();

    // Verify results
    auto& result = computeRegionVec[0];

    // Should have 2 address vectors (ping-pong)
    ASSERT_EQ(result.bufferAddressVec.first.size(), 2);   // 2 local buffers in first set
    ASSERT_EQ(result.bufferAddressVec.second.size(), 2);  // 2 local buffers in second set

    // Addresses in first set should not overlap with second set
    auto& firstSet = result.bufferAddressVec.first;
    auto& secondSet = result.bufferAddressVec.second;

    // First set starts from 0
    EXPECT_EQ(firstSet[0], 0);
    EXPECT_EQ(firstSet[1], 1024);  // After first buffer

    // Second set should be offset from first set
    EXPECT_GE(secondSet[0], 2048);  // After first set (2 * 1024)

    // Base alignment should be set
    EXPECT_EQ(result.baseAlignment, 64);

    // prefetchOpCount should be 1 (we can prefetch the next iteration)
    EXPECT_EQ(result.prefetchOpCount, 1);
}

TEST_F(MLIR_LoopAllocator, TilingUnevenSizes) {
    // Tiling scenario:
    // - 3 iterations
    // - Iteration 0 & 1: 1024 bytes each
    // - Iteration 2: 512 bytes (last tile is smaller)
    // The allocator should use max(1024, 1024, 512) = 1024 for allocation

    ComputeRegionVec computeRegionVec;

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    // Iteration 0: 1024 bytes
    LoopBody iteration0;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {inBuf}, {outBuf}));
    }

    // Iteration 1: 1024 bytes
    LoopBody iteration1;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {inBuf}, {outBuf}));
    }

    // Iteration 2: 512 bytes (smaller last tile)
    LoopBody iteration2;
    {
        auto inBuf = createBuffer(512, 64);
        auto outBuf = createBuffer(512, 64);
        iteration2.push_back(createComputeOp(2, {inBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2)};

    ComputeRegion region(std::move(schedulingLoop));
    computeRegionVec.push_back(std::move(region));

    vpux::AddressType memorySize = 16384;  // 16KB
    LoopAllocator allocator(computeRegionVec, memorySize, _log, "test");
    allocator.allocateLoopTilingRegions();

    auto& result = computeRegionVec[0];

    // Verify that allocation used max size (1024) not the smaller size (512)
    // First set: buf0 at 0, buf1 at 1024
    // Second set: buf0 at 2048, buf1 at 3072
    ASSERT_EQ(result.bufferAddressVec.first.size(), 2);
    EXPECT_EQ(result.bufferAddressVec.first[0], 0);
    EXPECT_EQ(result.bufferAddressVec.first[1], 1024);

    // The reservation size should account for max buffer sizes
    // 2 buffers * 1024 bytes * 2 sets = 4096 bytes
    EXPECT_EQ(result.size, 4096);
}

TEST_F(MLIR_LoopAllocator, TilingNoPrefetch) {
    // Case 1: No prefetching
    // CMX buffer only enough for the first iteration.
    //
    // ┌───────────────────────────┐ ← memorySize
    // │ Shared Buffers            │
    // ├───────────────────────────┤ ← firstSet.nextFreeOffset
    // │ Buffer 2 (1st set)        │ ← reuse 2nd set
    // │ Buffer 1 (1st set)        │ ← reuse 2nd set
    // └───────────────────────────┘ ← 0
    //
    // All tiles will reuse the same buffer.
    //
    // - 4 iterations
    // - 2 buffers per iteration, each 1024 bytes
    // - Memory size: 2048 bytes (only enough for 1st set)
    //
    // Expected: All iterations reuse the same addresses (no ping-pong)
    //   iter 0: uses [0, 1024]
    //   iter 1: reuses [0, 1024] (same as 1st set)
    //   iter 2: reuses [0, 1024]
    //   iter 3: reuses [0, 1024]

    ComputeRegionVec computeRegionVec;

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    // Iteration 0
    LoopBody iteration0;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {inBuf}, {outBuf}));
    }

    // Iteration 1
    LoopBody iteration1;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {inBuf}, {outBuf}));
    }

    // Iteration 2
    LoopBody iteration2;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration2.push_back(createComputeOp(2, {inBuf}, {outBuf}));
    }

    // Iteration 3
    LoopBody iteration3;
    {
        auto inBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration3.push_back(createComputeOp(3, {inBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2),
                                  std::move(iteration3)};

    ComputeRegion region(std::move(schedulingLoop));
    computeRegionVec.push_back(std::move(region));

    // Memory size: exactly 2048 bytes (just enough for 1st set)
    vpux::AddressType memorySize = 2048;
    LoopAllocator allocator(computeRegionVec, memorySize, _log, "test");
    allocator.allocateLoopTilingRegions();

    auto& result = computeRegionVec[0];

    ASSERT_EQ(result.bufferAddressVec.first.size(), 2);
    ASSERT_EQ(result.bufferAddressVec.second.size(), 2);

    auto& firstSet = result.bufferAddressVec.first;    // for even iterations (0, 2, ...)
    auto& secondSet = result.bufferAddressVec.second;  // for odd iterations (1, 3, ...)

    // First set
    EXPECT_EQ(firstSet[0], 0);
    EXPECT_EQ(firstSet[1], 1024);

    // Second set should fully reuse first set (no prefetch possible)
    // This means iter 0,1,2,3 all use the same addresses [0, 1024]
    EXPECT_EQ(secondSet[0], firstSet[0]);  // Reused
    EXPECT_EQ(secondSet[1], firstSet[1]);  // Reused

    // prefetchOpCount should be 0
    EXPECT_EQ(result.prefetchOpCount, 0);

    // reserved size should be 2048 (only 1st set)
    EXPECT_EQ(result.size, 2048);
}

TEST_F(MLIR_LoopAllocator, TilingSharedBufferWithPartialPrefetch) {
    // Case 2: Prefetching but no pipelining (Partial Pipeline)
    //
    // ┌───────────────────────────┐ ← memorySize
    // │ Shared Buffers            │
    // ├───────────────────────────┤
    // │ Buffer 2 (2nd set)        │ ← prefetch
    // │ Buffer 1 (2nd set)        │ ← prefetch
    // ├───────────────────────────┤ ← firstSet.nextFreeOffset
    // │ Buffer 3 (1st set)        │ ← reuse 2nd set
    // │ Buffer 2 (1st set)        │
    // │ Buffer 1 (1st set)        │
    // └───────────────────────────┘ ← 0
    //
    // All even (0, 2, 4...) tiles use 1st set buffer (1, 2, 3).
    // All odd (1, 3, 5...) tiles use 2nd set buffer (1, 2) and 1st set buffer (3).
    //
    // - 4 iterations to verify partial ping-pong reuse pattern
    // - weight (shared, 2048 bytes) - used by all iterations
    // - actIn0, actIn1, output (local, 1024 bytes each)
    //
    // Memory layout:
    //   sharedRegionSize = 2048
    //   Available for local = 7168 - 2048 = 5120
    //   1st set: 3 * 1024 = 3072
    //   2nd set: can fit 2 buffers (2048), OOM at 3rd
    //
    // Expected ping-pong with partial prefetch:
    //   1st set: [0, 1024, 2048]      - for even iterations
    //   2nd set: [3072, 4096, 2048]   - for odd iterations (output reuses 1st set)
    //
    //   iter 0 (even): actIn0=0, actIn1=1024, out=2048
    //   iter 1 (odd):  actIn0=3072, actIn1=4096, out=2048 (out reuses!)
    //   iter 2 (even): actIn0=0, actIn1=1024, out=2048
    //   iter 3 (odd):  actIn0=3072, actIn1=4096, out=2048 (out reuses!)

    ComputeRegionVec computeRegionVec;

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    // Shared weight buffer
    auto weightBuf = createBuffer(2048, 64);

    // Iteration 0 (even - uses 1st set)
    LoopBody iteration0;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto actIn1 = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {weightBuf, actIn0, actIn1}, {outBuf}));
    }

    // Iteration 1 (odd - uses 2nd set with partial reuse)
    LoopBody iteration1;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto actIn1 = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {weightBuf, actIn0, actIn1}, {outBuf}));
    }

    // Iteration 2 (even - reuses 1st set)
    LoopBody iteration2;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto actIn1 = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration2.push_back(createComputeOp(2, {weightBuf, actIn0, actIn1}, {outBuf}));
    }

    // Iteration 3 (odd - reuses 2nd set with partial reuse)
    LoopBody iteration3;
    {
        auto actIn0 = createBuffer(1024, 64);
        auto actIn1 = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration3.push_back(createComputeOp(3, {weightBuf, actIn0, actIn1}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2),
                                  std::move(iteration3)};

    ComputeRegion region(std::move(schedulingLoop));
    computeRegionVec.push_back(std::move(region));

    vpux::AddressType memorySize = 7168;
    LoopAllocator allocator(computeRegionVec, memorySize, _log, "test");
    allocator.allocateLoopTilingRegions();

    auto& result = computeRegionVec[0];

    // Verify shared buffer recognized
    EXPECT_EQ(result.sharedExternalBuffers.size(), 1);

    ASSERT_EQ(result.bufferAddressVec.first.size(), 3);
    ASSERT_EQ(result.bufferAddressVec.second.size(), 3);

    auto& firstSet = result.bufferAddressVec.first;    // for even iterations (0, 2)
    auto& secondSet = result.bufferAddressVec.second;  // for odd iterations (1, 3)

    // 1st set: sequential allocation for 3 local buffers
    EXPECT_EQ(firstSet[0], 0);     // actIn0
    EXPECT_EQ(firstSet[1], 1024);  // actIn1
    EXPECT_EQ(firstSet[2], 2048);  // output

    // 2nd set: first 2 prefetched, 3rd reuses from 1st set
    EXPECT_EQ(secondSet[0], 3072);  // actIn0 prefetched
    EXPECT_EQ(secondSet[1], 4096);  // actIn1 prefetched
    EXPECT_EQ(secondSet[2], 2048);  // output reuses 1st set (partial prefetch!)

    // prefetchOpCount = 0 (not all buffers were prefetched)
    EXPECT_EQ(result.prefetchOpCount, 0);

    // reserved size = 1st set (3072) + prefetched part of 2nd set (2048) = 5120
    EXPECT_EQ(result.size, 5120);
}

TEST_F(MLIR_LoopAllocator, TilingFullPipelining) {
    // Case 3: Full pipelining
    //
    // ┌───────────────────────────┐ ← memorySize
    // │ Shared Buffers            │
    // ├───────────────────────────┤
    // │ Buffer 2 (2nd set)        │ ← prefetch
    // │ Buffer 1 (2nd set)        │ ← prefetch
    // ├───────────────────────────┤ ← firstSet.nextFreeOffset
    // │ Buffer 2 (1st set)        │
    // │ Buffer 1 (1st set)        │
    // └───────────────────────────┘ ← 0
    //
    // All even (0, 2, 4...) tiles use 1st set buffer (1, 2).
    // All odd (1, 3, 5...) tiles use 2nd set buffer (1, 2).
    //
    // - 4 iterations to verify ping-pong address reuse pattern
    // - Buffer 0: weight (shared, used by all iterations)
    // - Buffer 1: activation input (different per iteration)
    // - Buffer 2: output (different per iteration)
    //
    // Expected ping-pong pattern:
    //   iter 0 (even): uses 1st set [0, 1024]
    //   iter 1 (odd):  uses 2nd set [2048, 3072]
    //   iter 2 (even): reuses 1st set [0, 1024]
    //   iter 3 (odd):  reuses 2nd set [2048, 3072]

    ComputeRegionVec computeRegionVec;

    auto schedulingLoop = std::make_unique<SchedulingLoop>();
    schedulingLoop->type = LoopType::Tiling;

    // Shared weight buffer
    auto weightBuf = createBuffer(2048, 64);

    // Iteration 0 (even - uses 1st set)
    LoopBody iteration0;
    {
        auto actInBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration0.push_back(createComputeOp(0, {weightBuf, actInBuf}, {outBuf}));
    }

    // Iteration 1 (odd - uses 2nd set)
    LoopBody iteration1;
    {
        auto actInBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration1.push_back(createComputeOp(1, {weightBuf, actInBuf}, {outBuf}));
    }

    // Iteration 2 (even - reuses 1st set)
    LoopBody iteration2;
    {
        auto actInBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration2.push_back(createComputeOp(2, {weightBuf, actInBuf}, {outBuf}));
    }

    // Iteration 3 (odd - reuses 2nd set)
    LoopBody iteration3;
    {
        auto actInBuf = createBuffer(1024, 64);
        auto outBuf = createBuffer(1024, 64);
        iteration3.push_back(createComputeOp(3, {weightBuf, actInBuf}, {outBuf}));
    }

    schedulingLoop->loopBodies = {std::move(iteration0), std::move(iteration1), std::move(iteration2),
                                  std::move(iteration3)};

    ComputeRegion region(std::move(schedulingLoop));
    computeRegionVec.push_back(std::move(region));

    vpux::AddressType memorySize = 16384;
    LoopAllocator allocator(computeRegionVec, memorySize, _log, "test");
    allocator.allocateLoopTilingRegions();

    auto& result = computeRegionVec[0];

    // Check that shared buffer is recognized
    EXPECT_EQ(result.sharedExternalBuffers.size(), 1);

    // Local buffers: actIn + out = 2 per iteration
    ASSERT_EQ(result.bufferAddressVec.first.size(), 2);
    ASSERT_EQ(result.bufferAddressVec.second.size(), 2);

    auto& firstSet = result.bufferAddressVec.first;    // for even iterations (0, 2, ...)
    auto& secondSet = result.bufferAddressVec.second;  // for odd iterations (1, 3, ...)

    // 1st set addresses (used by iter 0, 2, ...)
    EXPECT_EQ(firstSet[0], 0);     // actIn
    EXPECT_EQ(firstSet[1], 1024);  // out

    // 2nd set addresses (used by iter 1, 3, ...)
    EXPECT_EQ(secondSet[0], 2048);  // actIn
    EXPECT_EQ(secondSet[1], 3072);  // out

    // Full pipelining: prefetchOpCount = 1
    EXPECT_EQ(result.prefetchOpCount, 1);

    // Reserved size = 2 sets * 2 buffers * 1024 = 4096
    EXPECT_EQ(result.size, 4096);
}
