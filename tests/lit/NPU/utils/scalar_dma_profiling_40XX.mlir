//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env IE_NPU_LOG_FILTER=dump-statistics-of-task-ops vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --dump-statistics-of-task-ops -o /dev/null %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// This test verifies that scalar tensor DMAs (rank=0, e.g., memref<f16>) are handled correctly
// without crashing in the profiling code path. The reduceDimsForDma function should handle
// empty memShape/memStrides arrays gracefully.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @scalar_dma_test attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.MemoryResource 31457280 bytes of @DDR {config.bandwidth = 8, config.derateFactor = 6.000000e-01}
  config.Resources 1 of @NCE  {
    config.MemoryResource 2097152 bytes of @CMX_NN {config.bandwidth = 32, config.derateFactor = 1.000000e+00}
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 2 of @DMA_NN

  net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x16x4x4xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x16x4x4xf16>
    }

  func.func @main(%arg0: memref<1x16x4x4xf16, @DDR>, %arg1: memref<1x16x4x4xf16, @DDR>) -> memref<1x16x4x4xf16, @DDR> {
    // Scalar constant - this is a rank-0 tensor (memref<f16>)
    %cst_scalar = const.Declare memref<f16> = dense<1.0> : tensor<f16>

    %buf_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<f16, [@CMX_NN, 0]>

    %bar0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier

    // DMA transferring a scalar value - should not crash
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA <{port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 0 : i64>}>
        inputs(%cst_scalar : memref<f16>)
        outputs(%buf_cmx : memref<f16, [@CMX_NN, 0]>) -> memref<f16, [@CMX_NN, 0]>
    }

    return %arg1 : memref<1x16x4x4xf16, @DDR>
  }
}

// CHECK: VPUIP.NNDMA - 1 ops
