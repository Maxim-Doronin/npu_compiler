//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --setup-npu-constraint="workload-management-enable=true enable-sw-kernel-fifo-per-shave-engine=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @mainModule attributes { VPU.arch = #VPU.arch_kind<NPU40XX> } {
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @VPU.UseDedicatedFifoPerShaveEngine : true
// Currently, still non-WLM barrier configuration settings are present even for WLM enabled mode. To be updated to WLM values in E#155846
// CHECK: config.Option @VPU.BarrierMaxVariantSum : 64
// CHECK: config.Option @VPU.BarrierMaxVariantCount : 128
// CHECK: config.Option @VPU.MetadataMaxVariantCount : 128
// CHECK: config.Option @VPU.MetadataMaxInvariantCount : 64
// CHECK: config.Option @VPU.MetadataMaxKernelInvocationCount : 32
// CHECK: config.Option @VPU.MetadataMaxKernelRangeCount : 32
