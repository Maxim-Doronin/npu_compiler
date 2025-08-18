//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --init-resources="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

// CHECK: module @mode attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<ReferenceSW>, config.revisionID = #config.revision_id<REVISION_NONE>}
module @mode attributes {config.compilationMode = #config.compilation_mode<ReferenceSW>} {
}

// -----

// CHECK: module @arch attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>, config.revisionID = #config.revision_id<REVISION_NONE>}
module @arch attributes {config.arch = #config.arch_kind<NPU37XX>} {
}

// -----

// CHECK: module @executors attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>, config.revisionID = #config.revision_id<REVISION_NONE>}
module @executors {
    IE.ExecutorResource 5 of @DMA_NN
    IE.TileResource 5 of @NCE at 6.000000e+02 MHz
}

// CHECK-DAG:    {{  }}IE.ExecutorResource 5 of @DMA_NN
// CHECK-DAG:    {{  }}IE.TileResource 5 of @NCE at 6.000000e+02 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @SHAVE_NN
// CHECK-DAG:    {{    }}IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}


// -----

// CHECK: module @memory attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>, config.revisionID = #config.revision_id<REVISION_NONE>}
module @memory {
    IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 5 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 10000 bytes of @CMX_NN {config.bandwidth = 10 : i64, config.derateFactor = 2.0 : f64}
    }
    IE.MemoryResource 500000 bytes of @DDR
}

// CHECK-DAG:    {{  }}IE.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:    {{  }}IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @SHAVE_NN
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}IE.MemoryResource 5 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 10000 bytes of @CMX_NN {config.bandwidth = 10 : i64, config.derateFactor = 2.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 500000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
