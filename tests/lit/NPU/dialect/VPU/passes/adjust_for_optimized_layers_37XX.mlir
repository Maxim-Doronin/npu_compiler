//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-for-optimized-layers %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustForSoftmaxMultiShaveOptNCHW
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x2x16x32xf16>)
func.func @AdjustForSoftmaxMultiShaveOptNCHW(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>
    return %0 : tensor<1x2x16x32xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 4, 8, 32]} inputs([[ARG_0]] : tensor<1x2x16x32xf16>) -> tensor<1x4x8x32xf16>
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x4x8x32xf16> -> tensor<1x4x8x32xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [1, 2, 16, 32]} inputs([[SOFTMAX]] : tensor<1x4x8x32xf16>) -> tensor<1x2x16x32xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL: @AdjustForSoftmaxMultiShaveOptNHWC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x2x16xf16, {order = #NHWC}>)
func.func @AdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>
    return %0 : tensor<1x32x2x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 32, 4, 8]} inputs([[ARG_0]] : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x4x8xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 1 : i64} : tensor<1x32x4x8xf16, {order = #NHWC}> -> tensor<1x32x4x8xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 2, 16]} inputs([[SOFTMAX]] : tensor<1x32x4x8xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL: @NotAdjustForSoftmaxMultiShaveOptNHWC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x3x3xf16, {order = #NHWC}>)
func.func @NotAdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x64x3x3xf16, {order = #NHWC}>) -> tensor<1x64x3x3xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x64x3x3xf16, {order = #NHWC}> -> tensor<1x64x3x3xf16, {order = #NHWC}>
    return %0 : tensor<1x64x3x3xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG_0]]) {axisInd = 1 : i64} : tensor<1x64x3x3xf16, {order = #NHWC}> -> tensor<1x64x3x3xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL: @AdjustForSoftmaxMultiShaveOptNCHWwithBatch
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<2x4x16x16xf16>)
func.func @AdjustForSoftmaxMultiShaveOptNCHWwithBatch(%arg0: tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<2x4x16x16xf16> -> tensor<2x4x16x16xf16>
    return %0 : tensor<2x4x16x16xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 8, 16, 16]} inputs([[ARG_0]] : tensor<2x4x16x16xf16>) -> tensor<1x8x16x16xf16>
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x8x16x16xf16> -> tensor<1x8x16x16xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [2, 4, 16, 16]} inputs([[SOFTMAX]] : tensor<1x8x16x16xf16>) -> tensor<2x4x16x16xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL: @AdjustForSoftmaxMultiShaveOptNHWCwithBatch
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<2x4x16x16xf16, {order = #NHWC}>)
func.func @AdjustForSoftmaxMultiShaveOptNHWCwithBatch(%arg0: tensor<2x4x16x16xf16, {order = #NHWC}>) -> tensor<2x4x16x16xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<2x4x16x16xf16, {order = #NHWC}> -> tensor<2x4x16x16xf16, {order = #NHWC}>
    return %0 : tensor<2x4x16x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 4, 32, 16]} inputs([[ARG_0]] : tensor<2x4x16x16xf16, {order = #NHWC}>) -> tensor<1x4x32x16xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x4x32x16xf16, {order = #NHWC}> -> tensor<1x4x32x16xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [2, 4, 16, 16]} inputs([[SOFTMAX]] : tensor<1x4x32x16xf16, {order = #NHWC}>) -> tensor<2x4x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}
