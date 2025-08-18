//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --split-enqueue-dma-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @DpuEnqueueDmaNoNeedSplit attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 2 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <3, -1> -> !VPURegMapped.Index<0:0:1>
    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %enq_dma = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    %19 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %25 = VPURegMapped.ViewTaskRange(%19 -> %20 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8>
    %26 = VPURegMapped.ViewTaskRange(%15 -> %16 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8, [@CMX_NN, 0]>
    %35 = VPURegMapped.Enqueue (%enq_dma -> %enq_dma : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
    %37 = VPUMI40XX.MappedInference dmas((%enq_dma) : (!VPURegMapped.Index<0:0:0>)) invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%35 : !VPURegMapped.Index<0:0:0>) dmaCount([[1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(2) workItemCount(1) bootsrapWorkItemsCount(1) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[DMA_ENQ:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>, port = 0 : i64}
//CHECK-SAME: updates([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:0>
//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: ([[DMA_ENQ]] -> [[DMA_ENQ]] : <0:0:0> -> <0:0:0>)
//CHECK: dmas(([[DMA_ENQ]])
//CHECK-SAME: workItemCount(1)
//CHECK-SAME: bootsrapWorkItemsCount(1)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @DpuEnqueueDmaNeedSplit attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 2 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <3, -1> -> !VPURegMapped.Index<0:0:1>
    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %enq_dma = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    %19 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %25 = VPURegMapped.ViewTaskRange(%19 -> %20 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8>
    %26 = VPURegMapped.ViewTaskRange(%15 -> %16 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8, [@CMX_NN, 0]>
    %35 = VPURegMapped.Enqueue (%enq_dma -> %enq_dma : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
    %37 = VPUMI40XX.MappedInference dmas((%enq_dma) : (!VPURegMapped.Index<0:0:0>)) invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%35 : !VPURegMapped.Index<0:0:0>) dmaCount([[1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(2) workItemCount(1) bootsrapWorkItemsCount(1) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[DMA_ENQ0:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 0 : i64>, port = 0 : i64}
//CHECK-NOT: updates(
//CHECK-SAME: -> !VPURegMapped.Index<0:0:0>
//CHECK: [[DMA_ENQ1:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 1 : i64, endTask = 1 : i64>, port = 0 : i64}
//CHECK-SAME: previousDMA([[DMA_ENQ0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: updates([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:1>
//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: ([[DMA_ENQ0]] -> [[DMA_ENQ1]] : <0:0:0> -> <0:0:1>)
//CHECK: dmas(([[DMA_ENQ0]])
//CHECK-SAME: workItemCount(1)
//CHECK-SAME: bootsrapWorkItemsCount(1)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @DpuEnqueueDmaNeedSplitMultipleTimes attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 1 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 4 : ui8}(%11 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>

    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:2>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:3>

    %19 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %20 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %21 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:2>
    %22 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:3>

    %some_dma = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>

    %enq_dma = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 3 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    %23 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %24 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%23 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %25 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%17 : !VPURegMapped.Index<0:0:2>) previousTask(%24 : !VPURegMapped.Index<0:0:1>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:2> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %26 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%18 : !VPURegMapped.Index<0:0:3>) previousTask(%25 : !VPURegMapped.Index<0:0:2>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:3> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }

    %27 = VPUMI40XX.DPUVariant taskLocation(%19 : !VPURegMapped.Index<0:0:0>) calls(%23 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %28 = VPUMI40XX.DPUVariant taskLocation(%20 : !VPURegMapped.Index<0:0:1>) previousTask(%27 : !VPURegMapped.Index<0:0:0>) calls(%24 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %29 = VPUMI40XX.DPUVariant taskLocation(%21 : !VPURegMapped.Index<0:0:2>) previousTask(%28 : !VPURegMapped.Index<0:0:1>) calls(%25 : <0:0:2>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:2>
    %30 = VPUMI40XX.DPUVariant taskLocation(%22 : !VPURegMapped.Index<0:0:3>) previousTask(%29 : !VPURegMapped.Index<0:0:2>) calls(%26 : <0:0:3>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:3>

    %31 = VPURegMapped.Enqueue (%enq_dma -> %enq_dma : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
    %32 = VPURegMapped.Enqueue previousTaskIdx(%31 : !VPURegMapped.Index<0:0:0>) (%some_dma -> %some_dma : <1:0:0> -> <1:0:0>) -> !VPURegMapped.Index<1:0:0> {taskType = #VPURegMapped.task_type<DMA>}

    %33 = VPUMI40XX.MappedInference dmas((%enq_dma), (%some_dma) : (!VPURegMapped.Index<0:0:0>), (!VPURegMapped.Index<1:0:0>)) invariants(%23 : !VPURegMapped.Index<0:0:0>) variants(%27 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%31 : !VPURegMapped.Index<0:0:0>) dmaCount([[1],[1]]) invariantCount([4]) variantCount([4]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(3) workItemCount(2) bootsrapWorkItemsCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[BAR2:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[DMA:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {port = 1 : i64}
//CHECK-NOT: waits(
//CHECK-SAME: updates([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: -> !VPURegMapped.Index<1:0:0>
//CHECK: [[DMA_ENQ0:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 0 : i64>, port = 0 : i64}
//CHECK-SAME: waits([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-NOT: updates(
//CHECK-SAME: -> !VPURegMapped.Index<0:0:0>
//CHECK: [[DMA_ENQ1:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 1 : i64, endTask = 1 : i64>, port = 0 : i64}
//CHECK-SAME: previousDMA([[DMA_ENQ0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-NOT: waits(
//CHECK-NOT: updates(
//CHECK-SAME: -> !VPURegMapped.Index<0:0:1>
//CHECK: [[DMA_ENQ2:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 2 : i64, endTask = 2 : i64>, port = 0 : i64}
//CHECK-SAME: previousDMA([[DMA_ENQ1]] : !VPURegMapped.Index<0:0:1>)
//CHECK-NOT: waits(
//CHECK-NOT: updates(
//CHECK-SAME: -> !VPURegMapped.Index<0:0:2>
//CHECK: [[DMA_ENQ3:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 3 : i64, endTask = 3 : i64>, port = 0 : i64}
//CHECK-SAME: previousDMA([[DMA_ENQ2]] : !VPURegMapped.Index<0:0:2>)
//CHECK-NOT: waits(
//CHECK-SAME: updates([[BAR1]] : !VPURegMapped.Index<0:0:1>)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:3>
//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: ([[DMA_ENQ0]] -> [[DMA_ENQ3]] : <0:0:0> -> <0:0:3>)
//CHECK: dmas(([[DMA_ENQ0]])
//CHECK-SAME: workItemCount(2)
//CHECK-SAME: bootsrapWorkItemsCount(2

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @DpuEnqueueDmaNeedSplitMultipleTimes attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 2 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 4 : ui8}(%11 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>

    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:2>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:3>

    %19 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %20 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %21 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:2>
    %22 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:3>

    %some_dma = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    %enq_dma0 = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) previousDMA(%some_dma : !VPURegMapped.Index<0:0:0>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %enq_dma1 = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 2 : i64, endTask = 3 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) previousDMA(%enq_dma0 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>

    %23 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %24 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%23 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %25 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%17 : !VPURegMapped.Index<0:0:2>) previousTask(%24 : !VPURegMapped.Index<0:0:1>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:2> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %26 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%18 : !VPURegMapped.Index<0:0:3>) previousTask(%25 : !VPURegMapped.Index<0:0:2>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:3> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }

    %27 = VPUMI40XX.DPUVariant taskLocation(%19 : !VPURegMapped.Index<0:0:0>) calls(%23 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %28 = VPUMI40XX.DPUVariant taskLocation(%20 : !VPURegMapped.Index<0:0:1>) previousTask(%27 : !VPURegMapped.Index<0:0:0>) calls(%24 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %29 = VPUMI40XX.DPUVariant taskLocation(%21 : !VPURegMapped.Index<0:0:2>) previousTask(%28 : !VPURegMapped.Index<0:0:1>) calls(%25 : <0:0:2>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:2>
    %30 = VPUMI40XX.DPUVariant taskLocation(%22 : !VPURegMapped.Index<0:0:3>) previousTask(%29 : !VPURegMapped.Index<0:0:2>) calls(%26 : <0:0:3>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:3>

    %31 = VPURegMapped.Enqueue (%some_dma -> %enq_dma1 : <0:0:0> -> <0:0:2>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}

    %32 = VPUMI40XX.MappedInference dmas((%some_dma) : (!VPURegMapped.Index<0:0:0>)) invariants(%23 : !VPURegMapped.Index<0:0:0>) variants(%27 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%31 : !VPURegMapped.Index<0:0:0>) dmaCount([[3]]) invariantCount([4]) variantCount([4]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(3) workItemCount(1) bootsrapWorkItemsCount(1) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[BAR2:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK: [[DMA:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {port = 0 : i64}
//CHECK-NOT: waits(
//CHECK-SAME: updates([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:0>
//CHECK: [[DMA_ENQ0:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 0 : i64>, port = 0 : i64}
//CHECK-SAME: previousDMA([[DMA]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: waits([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-NOT: updates(
//CHECK-SAME: -> !VPURegMapped.Index<0:0:1>
//CHECK: [[DMA_ENQ1:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 1 : i64, endTask = 1 : i64>, port = 0 : i64}
//CHECK-SAME: previousDMA([[DMA_ENQ0]] : !VPURegMapped.Index<0:0:1>)
//CHECK-NOT: waits(
//CHECK-SAME: updates([[BAR1]] : !VPURegMapped.Index<0:0:1>)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:2>
//CHECK: [[DMA_ENQ2:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 2 : i64, endTask = 2 : i64>, port = 0 : i64}
//CHECK-SAME: previousDMA([[DMA_ENQ1]] : !VPURegMapped.Index<0:0:2>)
//CHECK-SAME: waits([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-NOT: updates(
//CHECK-SAME: -> !VPURegMapped.Index<0:0:3>
//CHECK: [[DMA_ENQ3:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 3 : i64, endTask = 3 : i64>, port = 0 : i64}
//CHECK-SAME: previousDMA([[DMA_ENQ2]] : !VPURegMapped.Index<0:0:3>)
//CHECK-NOT: waits(
//CHECK-SAME: updates([[BAR1]] : !VPURegMapped.Index<0:0:1>)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:4>
//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: ([[DMA]] -> [[DMA_ENQ3]] : <0:0:0> -> <0:0:4>)
//CHECK: dmas(([[DMA]])
//CHECK-SAME: workItemCount(1)
//CHECK-SAME: bootsrapWorkItemsCount(1)
