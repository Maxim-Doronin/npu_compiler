//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --setup-channels-auto-padding="enable-auto-padding-odu enable-auto-padding-idu" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @mainModule attributes {} {
}

// CHECK: module @mainModule
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @VPU.AutoPaddingODU : true
// CHECK: config.Option @VPU.AutoPaddingIDU : true
