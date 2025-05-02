//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --inline %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// Note: This test checks if UnifiedFuncInlinerInterface uses the fallback implementation.
module @FuncCallOpFallback {
    func.func private @foo(%arg: tensor<f32>) -> tensor<f32> {
        %0 = VPU.Add(%arg, %arg) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<f32>, tensor<f32> -> tensor<f32>
        return %0: tensor<f32>
    }

    // CHECK-NOT: func.func private @foo

    func.func @main(%arg: tensor<f32>) -> (tensor<f32>) {
        %0 = func.call @foo(%arg) : (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<f32>)
    // CHECK: [[ADD0:%.+]] = VPU.Add([[ARG]], [[ARG]])
    // CHECK: return [[ADD0]]
}
