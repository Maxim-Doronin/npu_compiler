//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/core/loop_allocator.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"

#include "common/utils.hpp"
#include "feasible_memory_scheduler_test_utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

// Run cmd: npuUnitTests --gtest_filter="MLIR_FeasibleMemorySchedulerLoop.*"

using MLIR_FeasibleMemorySchedulerLoop = MLIR_UnitBase;

// Verifies that input buffers for loop operations are correctly reused across iterations in an tiling loop.
// Example from a convolution tiled 5 over output channels:
// Op 3 (Tile 0): Input[0]=360448(589824B), Input[1]=0(32768B), Input[2]=950272(1024B)
// 				 + Added
// Op 5 (Tile 1): Input[0]=360448(589824B), Input[1]=950272(1024B), Input[2]=180224(32768B)
// 				 + Added
// Op 7 (Tile 2): Input[0]=360448(589824B), Input[1]=950272(1024B), Input[2]=0(32768B)
//               ✓ Checked - match tile 0
// Op 9 (Tile 3): Input[0]=360448(589824B), Input[1]=950272(1024B), Input[2]=180224(32768B)
//               ✓ Checked - matches tile 1
// Op 12 (Tile 4): Input[0]=950272(1024B), Input[1]=360448(589824B), Input[2]=0(32768B)
//               ✓ Checked - match tile 0 again
bool FeasibleMemorySchedulerTest::verifyTilingLoopInputAddress() const {
    auto log = Logger::global().nest("loop-allocator-test");
    llvm::DenseMap<size_t, llvm::DenseSet<vpux::AddressType>> inputAddresses;  // Map: bufferSize - {set of addresses}
    // Collect the buffer addresses from the first 2 loop operations
    // which should be the same for all iterations in tiling loop
    const auto opNum = _scheduler._scheduledOps.size();
    if (opNum < 2) {
        log.error("Not enough scheduled operations to verify loop input addresses");
        return false;
    }
    auto isComputeLoopOp = [&](const FeasibleMemoryScheduler::ScheduledOpInfo& op) -> bool {
        if (!op.isLoopOp()) {
            return false;
        }

        if (op.numOfInputResources() == 0 || op.numOfOutputResources() == 0) {
            return false;
        }
        return true;
    };

    // Collect all buffer addresses from loop operations
    size_t checkedLoopOpCount = 0;
    for (const auto& scheduledOp : _scheduler._scheduledOps) {
        if (!isComputeLoopOp(scheduledOp)) {
            continue;
        }
        log.trace("Op {0} ({1}), scheduledOp.numOfInputResources() {2}", scheduledOp.op_,
                  scheduledOp.opTypeName().str(), scheduledOp.numOfInputResources());
        if (checkedLoopOpCount < 2) {
            // Store the first 2 loop ops' addresses
            for (size_t i = 0; i < scheduledOp.numOfInputResources(); i++) {
                if (!scheduledOp.isActiveInputResource(i)) {
                    continue;
                }
                vpux::AddressType beginAddr = scheduledOp.beginInputResource(i);
                vpux::AddressType bufferSize = scheduledOp.endInputResource(i) - beginAddr;
                log.trace("[Add] Input[{0}]: begin={1}, size={2}", i, beginAddr, bufferSize);
                inputAddresses[bufferSize].insert(beginAddr);
            }
        } else {
            // Check if the addresses are reused from the first 2 loop ops
            for (size_t i = 0; i < scheduledOp.numOfInputResources(); i++) {
                if (!scheduledOp.isActiveInputResource(i)) {
                    continue;
                }
                vpux::AddressType beginAddr = scheduledOp.beginInputResource(i);
                vpux::AddressType bufferSize = scheduledOp.endInputResource(i) - beginAddr;
                log.trace("[Check] Input[{0}]: begin={1}, size={2}", i, beginAddr, bufferSize);
                if (inputAddresses.count(bufferSize)) {
                    if (llvm::find(inputAddresses[bufferSize], beginAddr) == inputAddresses[bufferSize].end()) {
                        log.error("Unexpected buffer address {0} for size {1}, expected one of {2}", beginAddr,
                                  bufferSize, llvm::to_vector(inputAddresses[bufferSize]));
                        return false;
                    }
                }
            }
        }
        ++checkedLoopOpCount;
    }
    return true;
}

TEST_F(MLIR_FeasibleMemorySchedulerLoop, ScheduleLoopRegion) {
    // A single convolutional layer with tiling over C [1, 5, 1, 1]
    constexpr llvm::StringLiteral inputIRPart1 = R"(

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!TotalInputType = !VPUIP.DistributedBuffer<1x480x88x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 480, 30, 27], [1, 480, 29, 27], [1, 480, 29, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 59, 0]], memory_shapes = [[1, 480, 30, 27], [1, 480, 29, 27], [1, 480, 29, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 59, 0]]}>
!SliceWeightsType = !VPUIP.DistributedBuffer<96x480x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[96, 480, 1, 1], [96, 480, 1, 1], [96, 480, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[96, 480, 1, 1], [96, 480, 1, 1], [96, 480, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!SliceWTType = !VPUIP.DistributedBuffer<96x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!SliceOutputType = !VPUIP.DistributedBuffer<1x96x88x27xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 30, 27], [1, 96, 29, 27], [1, 96, 29, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 59, 0]], memory_shapes = [[1, 96, 30, 27], [1, 96, 29, 27], [1, 96, 29, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 59, 0]]}>

module @model_name attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU5010>} {
func.func @main(%arg0: memref<1x88x27x480xf16, @DDR>, %arg1: memref<1x480x88x27xf16, @DDR>) -> memref<1x480x88x27xf16, @DDR> {
  %cst = const.Declare memref<96x480x1x1xf16, #NHWC> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_0 = const.Declare memref<96x480x1x1xf16, #NHWC> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[96, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_1 = const.Declare memref<96x480x1x1xf16, #NHWC> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[192, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_2 = const.Declare memref<96x480x1x1xf16, #NHWC> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[288, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_3 = const.Declare memref<96x480x1x1xf16, #NHWC> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[384, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_4 = const.Declare memref<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
  %0 = VPURT.AllocDistributed -> !TotalInputType
  %1 = VPURT.AllocDistributed -> !SliceWeightsType
  %2 = VPURT.AllocDistributed -> !SliceWTType
  %3 = VPURT.AllocDistributed -> !SliceOutputType
  %4 = VPURT.AllocDistributed -> !SliceWeightsType
  %5 = VPURT.AllocDistributed -> !SliceOutputType
  %6 = VPURT.AllocDistributed -> !SliceWeightsType
  %7 = VPURT.AllocDistributed -> !SliceOutputType
  %8 = VPURT.AllocDistributed -> !SliceWeightsType
  %9 = VPURT.AllocDistributed -> !SliceOutputType
  %10 = VPURT.AllocDistributed -> !SliceWeightsType
  %11 = VPURT.AllocDistributed -> !SliceOutputType
  %token, %bodyResults = async.execute -> !async.value<!TotalInputType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 32071 : i64} {
    %18 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%arg0 : memref<1x88x27x480xf16, @DDR>) -> memref<1x480x88x27xf16, #NHWC, @DDR>
    %19 = VPUIP.NNDMA  inputs(%18 : memref<1x480x88x27xf16, #NHWC, @DDR>) outputs(%0 : !TotalInputType) -> !TotalInputType
    async.yield %19 : !TotalInputType
  }
  %token_5, %bodyResults_6 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA  inputs(%cst : memref<96x480x1x1xf16, #NHWC>) outputs(%1 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_7, %bodyResults_8 = async.execute -> !async.value<!SliceWTType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 607 : i64} {
    %18 = VPUIP.NNDMA  inputs(%cst_4 : memref<96x1x1x4xsi32>) outputs(%2 : !SliceWTType) -> !SliceWTType
    async.yield %18 : !SliceWTType
  }
  %token_9, %bodyResults_10 = async.execute [%token, %token_5, %token_7] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_6 as %arg3: !async.value<!SliceWeightsType>, %bodyResults_8 as %arg4: !async.value<!SliceWTType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 3 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg3 : !SliceWeightsType) weight_table(%arg4 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%3 : !SliceOutputType) outputs(%3 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_11, %bodyResults_12 = async.execute [%token_9] (%bodyResults_10 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA  inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  }
  %token_13, %bodyResults_14 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 5 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA  inputs(%cst_0 : memref<96x480x1x1xf16, #NHWC>) outputs(%4 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_15, %bodyResults_16 = async.execute [%token, %token_7, %token_13] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_8 as %arg3: !async.value<!SliceWTType>, %bodyResults_14 as %arg4: !async.value<!SliceWeightsType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 6 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg4 : !SliceWeightsType) weight_table(%arg3 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%5 : !SliceOutputType) outputs(%5 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_17, %bodyResults_18 = async.execute [%token_15] (%bodyResults_16 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 7 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 96, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA  inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  }
  %token_19, %bodyResults_20 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 8 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA  inputs(%cst_1 : memref<96x480x1x1xf16, #NHWC>) outputs(%6 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_21, %bodyResults_22 = async.execute [%token, %token_7, %token_19] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_8 as %arg3: !async.value<!SliceWTType>, %bodyResults_20 as %arg4: !async.value<!SliceWeightsType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 9 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg4 : !SliceWeightsType) weight_table(%arg3 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%7 : !SliceOutputType) outputs(%7 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_23, %bodyResults_24 = async.execute [%token_21] (%bodyResults_22 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 10 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 192, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA  inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  })";
    constexpr llvm::StringLiteral inputIRPart2 = R"(
  %token_25, %bodyResults_26 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 11 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA  inputs(%cst_2 : memref<96x480x1x1xf16, #NHWC>) outputs(%8 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_27, %bodyResults_28 = async.execute [%token, %token_7, %token_25] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_8 as %arg3: !async.value<!SliceWTType>, %bodyResults_26 as %arg4: !async.value<!SliceWeightsType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 12 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg4 : !SliceWeightsType) weight_table(%arg3 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%9 : !SliceOutputType) outputs(%9 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_29, %bodyResults_30 = async.execute [%token_27] (%bodyResults_28 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 13 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 288, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA  inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  }
  %token_31, %bodyResults_32 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 14 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA  inputs(%cst_3 : memref<96x480x1x1xf16, #NHWC>) outputs(%10 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_33, %bodyResults_34 = async.execute [%token, %token_7, %token_31] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_8 as %arg3: !async.value<!SliceWTType>, %bodyResults_32 as %arg4: !async.value<!SliceWeightsType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 15 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg4 : !SliceWeightsType) weight_table(%arg3 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%11 : !SliceOutputType) outputs(%11 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_35, %bodyResults_36 = async.execute [%token_33] (%bodyResults_34 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 16 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 384, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA  inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  }
  %12 = async.await %bodyResults_12 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %13 = async.await %bodyResults_18 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %14 = async.await %bodyResults_24 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %15 = async.await %bodyResults_30 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %16 = async.await %bodyResults_36 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %17 = VPUIP.ConcatView inputs(%12, %13, %14, %15, %16 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>, memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>, memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>, memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>, memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) outputs(%arg1 : memref<1x480x88x27xf16, @DDR>) -> memref<1x480x88x27xf16, @DDR>
  return %17 : memref<1x480x88x27xf16, @DDR>
}
})";

    const auto arch = config::ArchKind::NPU50XX;
    VPU::initializeSingletons(registry, VPU::DeviceVersion{std::nullopt, arch});

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>((inputIRPart1 + inputIRPart2).str(), &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    // Initialize the required components for scheduling
    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    auto liveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
    AsyncDepsInfo depsInfo{func};

    const auto memKind = VPU::MemoryKind::CMX_NN;
    const auto secondLvlMemKind = VPU::MemoryKind::DDR;
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    const auto availableCMXSize = 1473536;
    LinearScan<mlir::Value, LinearScanHandler> scan(availableCMXSize, {}, alignment);

    const auto vpuDevice = VPUNN::VPUDevice::NPU_5_0;
    auto costModel = VPU::CostModelConfig::createCostModel(&ctx);
    const int64_t nceClusterCount = 3;
    const int64_t dmaCount = 2;
    const bool enableScheduleStatistics = false;
    const bool optimizeFragmentation = false;
    const bool activelySpillForPrefetching = false;

    // Create FeasibleMemoryScheduler with empty loop regions for testing
    FeasibleMemoryScheduler noLoopRegionScheduler(
            memKind, secondLvlMemKind, liveRangeInfo, depsInfo, log, scan, arch, vpuDevice, costModel, nceClusterCount,
            dmaCount, enableScheduleStatistics, optimizeFragmentation, activelySpillForPrefetching,
            {}  // loopRegions will be empty for this test
    );

    // Create test fixture to access private members
    FeasibleMemorySchedulerTest testAccessor(noLoopRegionScheduler);

    // Loop region should be empty when no loopRegions is provided for scheduler
    EXPECT_EQ(testAccessor.getLoopRegionSize(), 0);
    // Scheduled loop regions should be zero as scheduling is not performed here
    EXPECT_EQ(testAccessor.getScheduledLoopRegionSize(), 0);

    // Create loop regions from tiling
    // And provide the loop for scheduler
    depsInfo = AsyncDepsInfo{func};
    depsInfo.buildConsMap();
    auto computeRegionVec = getComputeRegionsFromAsyncExec(aliasesInfo, depsInfo, log);
    // Include 2 LoopType::None regions and 1 tiling loop region with 5 loop bodies
    EXPECT_EQ(computeRegionVec.size(), 3);
    for (auto& region : computeRegionVec) {
        if (region.getLoopType() != LoopType::None) {
            EXPECT_EQ(region.schedulingLoop->loopBodies.size(), 5);
        }
    }
    LoopAllocator loopAllocator(computeRegionVec, availableCMXSize, log, "loop-allocator");
    loopAllocator.allocateLoopTilingRegions();
    FeasibleMemoryScheduler fiveTilesLoopRegionScheduler(memKind, secondLvlMemKind, liveRangeInfo, depsInfo, log, scan,
                                                         arch, vpuDevice, costModel, nceClusterCount, dmaCount,
                                                         enableScheduleStatistics, optimizeFragmentation,
                                                         activelySpillForPrefetching, std::move(computeRegionVec));
    FeasibleMemorySchedulerTest testAccessor2(fiveTilesLoopRegionScheduler);
    EXPECT_EQ(testAccessor2.getLoopRegionSize(), 3);
    fiveTilesLoopRegionScheduler.generateSchedule();
    // Scheduled loop regions should not be zero as scheduler has loop region
    EXPECT_EQ(testAccessor2.getScheduledLoopRegionSize(), 1);
    EXPECT_TRUE(testAccessor2.verifyTilingLoopInputAddress());
}

TEST_F(MLIR_FeasibleMemorySchedulerLoop, ScheduleLoopRegionTilingCH) {
    // A single convolutional layer with tiling over C and H [1, 2, 4, 1]
    // This creates a 2x4 tiling pattern (2 tiles in C dimension, 4 tiles in H dimension)
    constexpr llvm::StringLiteral inputIRPart1 = R"(

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @model_name attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU5010>} {
func.func @main(%arg0: memref<1x48x48x256xf16, @DDR>, %arg1: memref<1x256x48x48xf16, @DDR>) -> memref<1x256x48x48xf16, @DDR> {
  %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
  %cst_0 = const.Declare memref<64x256x1x1xf16, #NHWC> = dense<1.0> : tensor<256x256x1x1xf32>, [#const.SubView<[192, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_1 = const.Declare memref<64x256x1x1xf16, #NHWC> = dense<1.0> : tensor<256x256x1x1xf32>, [#const.SubView<[128, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_2 = const.Declare memref<64x256x1x1xf16, #NHWC> = dense<1.0> : tensor<256x256x1x1xf32>, [#const.SubView<[64, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_3 = const.Declare memref<64x256x1x1xf16, #NHWC> = dense<1.0> : tensor<256x256x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %alloc = memref.alloc() : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_4 = memref.alloc() : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_5 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
  %alloc_6 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_7 = memref.alloc() : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_8 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_9 = memref.alloc() : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_10 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_11 = memref.alloc() : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_12 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_13 = memref.alloc() : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_14 = memref.alloc() : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_15 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_16 = memref.alloc() : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_17 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_18 = memref.alloc() : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_19 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_20 = memref.alloc() : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %alloc_21 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %token, %bodyResults = async.execute -> !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 16718 : i64} {
    %9 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%arg0 : memref<1x48x48x256xf16, @DDR>) -> memref<1x256x48x48xf16, #NHWC, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 0, 0] [1, 256, 24, 48] : memref<1x256x48x48xf16, #NHWC, @DDR> to memref<1x256x24x48xf16, {order = #NHWC, strides = [589824, 1, 12288, 256]}, @DDR>
    %11 = VPUIP.NNDMA  inputs(%10 : memref<1x256x24x48xf16, {order = #NHWC, strides = [589824, 1, 12288, 256]}, @DDR>) outputs(%alloc : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %11 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_22, %bodyResults_23 = async.execute -> !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 1482 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst_3 : memref<64x256x1x1xf16, #NHWC>) outputs(%alloc_4 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_24, %bodyResults_25 = async.execute -> !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 600 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst : memref<64x1x1x4xsi32>) outputs(%alloc_5 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    async.yield %9 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
  }
  %token_26, %bodyResults_27 = async.execute [%token, %token_22, %token_24] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_23 as %arg3: !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_25 as %arg4: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 3 : i64, cycleCost = 25361 : i64} {
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg3 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg4 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_6 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_6 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_28, %bodyResults_29 = async.execute -> !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 1482 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst_2 : memref<64x256x1x1xf16, #NHWC>) outputs(%alloc_7 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_30, %bodyResults_31 = async.execute [%token, %token_24, %token_28] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_25 as %arg3: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_29 as %arg4: !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleCost = 25361 : i64} {
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_8 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_32, %bodyResults_33 = async.execute -> !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 6 : i64, cycleCost = 1482 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst_1 : memref<64x256x1x1xf16, #NHWC>) outputs(%alloc_9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_34, %bodyResults_35 = async.execute [%token, %token_24, %token_32] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_25 as %arg3: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_33 as %arg4: !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 7 : i64, cycleCost = 25361 : i64} {
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_10 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_10 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_36, %bodyResults_37 = async.execute -> !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 8 : i64, cycleCost = 1482 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst_0 : memref<64x256x1x1xf16, #NHWC>) outputs(%alloc_11 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_38, %bodyResults_39 = async.execute [%token, %token_24, %token_36] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_25 as %arg3: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_37 as %arg4: !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 9 : i64, cycleCost = 25361 : i64} {
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_12 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_12 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  })";

    constexpr llvm::StringLiteral inputIRPart2 = R"(
  %token_40, %bodyResults_41 = async.execute -> !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 10 : i64, cycleCost = 16718 : i64} {
    %9 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%arg0 : memref<1x48x48x256xf16, @DDR>) -> memref<1x256x48x48xf16, #NHWC, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 24, 0] [1, 256, 24, 48] : memref<1x256x48x48xf16, #NHWC, @DDR> to memref<1x256x24x48xf16, {order = #NHWC, strides = [589824, 1, 12288, 256]}, @DDR>
    %11 = VPUIP.NNDMA  inputs(%10 : memref<1x256x24x48xf16, {order = #NHWC, strides = [589824, 1, 12288, 256]}, @DDR>) outputs(%alloc_13 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %11 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_42, %bodyResults_43 = async.execute -> !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 11 : i64, cycleCost = 1482 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst_3 : memref<64x256x1x1xf16, #NHWC>) outputs(%alloc_14 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_44, %bodyResults_45 = async.execute [%token_24, %token_40, %token_42] (%bodyResults_25 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_41 as %arg3: !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_43 as %arg4: !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 12 : i64, cycleCost = 25361 : i64} {
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_15 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_15 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_46, %bodyResults_47 = async.execute -> !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 13 : i64, cycleCost = 1482 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst_2 : memref<64x256x1x1xf16, #NHWC>) outputs(%alloc_16 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_48, %bodyResults_49 = async.execute [%token_24, %token_40, %token_46] (%bodyResults_25 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_41 as %arg3: !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_47 as %arg4: !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 14 : i64, cycleCost = 25361 : i64} {
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_17 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_17 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_50, %bodyResults_51 = async.execute -> !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 15 : i64, cycleCost = 1482 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst_1 : memref<64x256x1x1xf16, #NHWC>) outputs(%alloc_18 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_52, %bodyResults_53 = async.execute [%token_24, %token_40, %token_50] (%bodyResults_25 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_41 as %arg3: !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_51 as %arg4: !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 16 : i64, cycleCost = 25361 : i64} {
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_19 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_19 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_54, %bodyResults_55 = async.execute -> !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 17 : i64, cycleCost = 1482 : i64} {
    %9 = VPUIP.NNDMA  inputs(%cst_0 : memref<64x256x1x1xf16, #NHWC>) outputs(%alloc_20 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
    async.yield %9 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  %token_56, %bodyResults_57 = async.execute [%token_24, %token_40, %token_54] (%bodyResults_25 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_41 as %arg3: !async.value<memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_55 as %arg4: !async.value<memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 18 : i64, cycleCost = 25361 : i64} {
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_21 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_21 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_58, %bodyResults_59 = async.execute [%token_26] (%bodyResults_27 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 19 : i64, cycleCost = 4619 : i64} {
    %9 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 64, 24, 48] : memref<1x256x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    %10 = VPUIP.NNDMA  inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    async.yield %10 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
  })";

    constexpr llvm::StringLiteral inputIRPart3 = R"(
  %token_60, %bodyResults_61 = async.execute [%token_30] (%bodyResults_31 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 20 : i64, cycleCost = 4619 : i64} {
    %9 = VPUIP.SubView %arg1 [0, 64, 0, 0] [1, 64, 24, 48] : memref<1x256x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    %10 = VPUIP.NNDMA  inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    async.yield %10 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
  }
  %token_62, %bodyResults_63 = async.execute [%token_34] (%bodyResults_35 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 21 : i64, cycleCost = 4619 : i64} {
    %9 = VPUIP.SubView %arg1 [0, 128, 0, 0] [1, 64, 24, 48] : memref<1x256x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    %10 = VPUIP.NNDMA  inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    async.yield %10 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
  }
  %token_64, %bodyResults_65 = async.execute [%token_38] (%bodyResults_39 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 22 : i64, cycleCost = 4619 : i64} {
    %9 = VPUIP.SubView %arg1 [0, 192, 0, 0] [1, 64, 24, 48] : memref<1x256x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    %10 = VPUIP.NNDMA  inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    async.yield %10 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
  }
  %token_66, %bodyResults_67 = async.execute [%token_44] (%bodyResults_45 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 23 : i64, cycleCost = 4619 : i64} {
    %9 = VPUIP.SubView %arg1 [0, 0, 24, 0] [1, 64, 24, 48] : memref<1x256x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    %10 = VPUIP.NNDMA  inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    async.yield %10 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
  }
  %token_68, %bodyResults_69 = async.execute [%token_48] (%bodyResults_49 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 24 : i64, cycleCost = 4619 : i64} {
    %9 = VPUIP.SubView %arg1 [0, 64, 24, 0] [1, 64, 24, 48] : memref<1x256x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    %10 = VPUIP.NNDMA  inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    async.yield %10 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
  }
  %token_70, %bodyResults_71 = async.execute [%token_52] (%bodyResults_53 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 25 : i64, cycleCost = 4619 : i64} {
    %9 = VPUIP.SubView %arg1 [0, 128, 24, 0] [1, 64, 24, 48] : memref<1x256x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    %10 = VPUIP.NNDMA  inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    async.yield %10 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
  }
  %token_72, %bodyResults_73 = async.execute [%token_56] (%bodyResults_57 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 26 : i64, cycleCost = 4619 : i64} {
    %9 = VPUIP.SubView %arg1 [0, 192, 24, 0] [1, 64, 24, 48] : memref<1x256x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    %10 = VPUIP.NNDMA  inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
    async.yield %10 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>
  }
  %0 = async.await %bodyResults_59 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>>
  %1 = async.await %bodyResults_61 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>>
  %2 = async.await %bodyResults_63 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>>
  %3 = async.await %bodyResults_65 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>>
  %4 = async.await %bodyResults_67 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>>
  %5 = async.await %bodyResults_69 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>>
  %6 = async.await %bodyResults_71 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>>
  %7 = async.await %bodyResults_73 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>>
  %8 = VPUIP.ConcatView inputs(%0, %1, %2, %3, %4, %5, %6, %7 : memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [589824, 2304, 48, 1]}, @DDR>) outputs(%arg1 : memref<1x256x48x48xf16, @DDR>) -> memref<1x256x48x48xf16, @DDR>
  return %8 : memref<1x256x48x48xf16, @DDR>
}
})";

    const auto arch = config::ArchKind::NPU50XX;
    VPU::initializeSingletons(registry, VPU::DeviceVersion{std::nullopt, arch});

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>((inputIRPart1 + inputIRPart2 + inputIRPart3).str(), &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    // Initialize the required components for scheduling
    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    auto liveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
    AsyncDepsInfo depsInfo{func};

    const auto memKind = VPU::MemoryKind::CMX_NN;
    const auto secondLvlMemKind = VPU::MemoryKind::DDR;
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    const auto availableCMXSize = 1473536;
    LinearScan<mlir::Value, LinearScanHandler> scan(availableCMXSize, {}, alignment);

    const auto vpuDevice = VPUNN::VPUDevice::NPU_5_0;
    auto costModel = VPU::CostModelConfig::createCostModel(&ctx);
    const int64_t nceClusterCount = 3;
    const int64_t dmaCount = 2;
    const bool enableScheduleStatistics = false;
    const bool optimizeFragmentation = false;
    const bool activelySpillForPrefetching = false;

    // Create loop regions from tiling over C and H [1, 2, 4, 1]
    depsInfo.buildConsMap();
    auto computeRegionVec = getComputeRegionsFromAsyncExec(aliasesInfo, depsInfo, log);
    // Expect 5 regions: 3 non-loop regions for global input DMA + 2 loop region with 4 tiles
    EXPECT_EQ(computeRegionVec.size(), 5);
    for (auto& region : computeRegionVec) {
        if (region.getLoopType() != LoopType::None) {
            // 2 tiles in C dimension * 2 tiles in H dimension = 4 total tiles
            EXPECT_EQ(region.schedulingLoop->loopBodies.size(), 4);
        }
    }

    LoopAllocator loopAllocator(computeRegionVec, availableCMXSize, log, "loop-allocator");
    loopAllocator.allocateLoopTilingRegions();

    FeasibleMemoryScheduler fourTilesLoopRegionScheduler(memKind, secondLvlMemKind, liveRangeInfo, depsInfo, log, scan,
                                                         arch, vpuDevice, costModel, nceClusterCount, dmaCount,
                                                         enableScheduleStatistics, optimizeFragmentation,
                                                         activelySpillForPrefetching, std::move(computeRegionVec));
    FeasibleMemorySchedulerTest testAccessor(fourTilesLoopRegionScheduler);
    EXPECT_EQ(testAccessor.getLoopRegionSize(), 5);
    fourTilesLoopRegionScheduler.generateSchedule();
    // Scheduled loop regions should not be zero as scheduler has loop region
    EXPECT_EQ(testAccessor.getScheduledLoopRegionSize(), 2);
    EXPECT_TRUE(testAccessor.verifyTilingLoopInputAddress());
}
