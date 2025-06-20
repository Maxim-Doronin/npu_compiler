//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

// CHECK: module @test attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<ReferenceSW>}
module @test {

// CHECK-DAG:    {{  }}config.PipelineOptions @Options {
// CHECK-DAG:    {{    }}config.Option @VPU.BarrierMaxVariantSum : 256
// CHECK-DAG:    {{    }}config.Option @VPU.BarrierMaxVariantCount : 256
// CHECK-DAG:    {{    }}config.Option @VPU.AutoPaddingODU : false
// CHECK-DAG:    {{    }}config.Option @VPU.AutoPaddingIDU : false
// CHECK-DAG:    {{    }}config.Option @VPU.MaxKernelSize : 11
// CHECK-DAG:    {{    }}config.Option @VPU.FragmentationAvoidRatioPipeliningLargeWeights : 4.500000e-01 : f32

// CHECK-DAG:    {{  }}}

// CHECK-DAG:    {{  }}IE.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:    {{  }}IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @SHAVE_NN
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 67108864000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

}
