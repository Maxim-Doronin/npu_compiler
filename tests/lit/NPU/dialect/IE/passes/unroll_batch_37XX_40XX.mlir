//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-batch %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

// CHECK-LABEL: @NotUnrollSubtract
// CHECK-SAME:  [[IN1:%.+]]: tensor<2x3x576x576xf16>
// CHECK-SAME:  [[IN2:%.+]]: tensor<2x1x576x576xf16>
func.func @NotUnrollSubtract(%input1 : tensor<2x3x576x576xf16>, %input2 : tensor<2x1x576x576xf16>) -> tensor<2x3x576x576xf16> {
    %0 = IE.Subtract(%input1, %input2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
                     : tensor<2x3x576x576xf16>, tensor<2x1x576x576xf16> -> tensor<2x3x576x576xf16>
    return %0 : tensor<2x3x576x576xf16>

    // CHECK: [[SUB:%.+]] = IE.Subtract([[IN1]], [[IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:          tensor<2x3x576x576xf16>, tensor<2x1x576x576xf16> -> tensor<2x3x576x576xf16>
    // CHECK-NOT:  IE.Slice
    // CHECK-NOT:  IE.Concat
    // CHECK:   return [[SUB]] : tensor<2x3x576x576xf16>
}
