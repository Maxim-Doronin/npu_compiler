//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=NPU50XX compilation-mode=ReferenceSW allow-custom-values=true available-cmx-memory=1997824" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU50XX

module @test attributes {config.platform=#config.platform<NPU5020>} {

// CHECK: module @test attributes {config.compilationMode = #config.compilation_mode<ReferenceSW>, config.elf_version = #config.version<{{[^>]*}}>, config.platform = #config.platform<NPU5020>, config.revisionID = #config.revision_id<REVISION_NONE>}

// CHECK-DAG:    {{  }}config.PipelineOptions @Options {
// CHECK-DAG:    {{    }}config.Option @config.BarrierMaxVariantSum : 64
// CHECK-DAG:    {{    }}config.Option @config.BarrierMaxVariantCount : 128
// CHECK-DAG:    {{    }}config.Option @config.AutoPaddingODU : false
// CHECK-DAG:    {{    }}config.Option @config.AutoPaddingIDU : false
// CHECK-DAG:    {{    }}config.Option @config.MaxKernelSize : 15
// CHECK-DAG:    {{    }}config.Option @config.FragmentationAvoidRatioPipeliningLargeWeights : 3.200000e-01 : f32
// CHECK-DAG:    {{  }}}

// CHECK-DAG:    {{  }}config.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:    {{  }}config.ExecutorResource 1 of @M2I
// CHECK-DAG:    {{  }}config.Resources 3 of @NCE at 2.100000e+03 MHz {
// CHECK-DAG:    {{    }}config.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}config.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}config.MemoryResource 1798041 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:   {{  }}config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}

}
