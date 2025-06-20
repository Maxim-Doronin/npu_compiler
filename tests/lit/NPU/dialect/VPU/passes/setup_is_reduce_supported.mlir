//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --setup-is-reduce-supported="enable-is-reduce-supported" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @mainModule attributes {} {
}

// CHECK: module @mainModule
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @VPU.ReduceSupported : true
