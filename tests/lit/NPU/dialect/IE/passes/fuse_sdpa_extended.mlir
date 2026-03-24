//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-sdpa-extended --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @Fuse_MM_SM_MM
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x24x225x16xf16>, [[ARG1:%.+]]: tensor<1x24x225x16xf16>, [[ARG2:%.+]]: tensor<1x24x225x16xf16>)
func.func @Fuse_MM_SM_MM(%arg0: tensor<1x24x225x16xf16>, %arg1: tensor<1x24x225x16xf16>, %arg2: tensor<1x24x225x16xf16>) -> tensor<1x24x225x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16> -> tensor<1x24x225x225xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x24x225x225xf16> -> tensor<1x24x225x225xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x24x225x16xf16> -> tensor<1x24x16x225xf16>
  %3 = IE.MatMul(%1, %2) {transpose_b} : tensor<1x24x225x225xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
  return %3 : tensor<1x24x225x16xf16>

  //CHECK: [[TRANS:%.+]] = IE.Transpose
  //CHECK: IE.SDPAExtended([[ARG0]], [[ARG1]], [[TRANS]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
}

// -----

// CHECK-LABEL: @Fuse_SDPA
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x55x128xf32>, [[ARG1:%.+]]: tensor<1x1x80x128xf32>, [[ARG2:%.+]]: tensor<1x1x80x128xf32>)
func.func @Fuse_SDPA(%arg0: tensor<1x1x55x128xf32>, %arg1: tensor<1x1x80x128xf32>, %arg2: tensor<1x1x80x128xf32>) -> tensor<1x1x55x128xf32> {
  %cst = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>} : tensor<1x1x55x128xf32>, tensor<1x1x80x128xf32>, tensor<1x1x80x128xf32>, tensor<1xf32> -> tensor<1x1x55x128xf32>
  return %0 : tensor<1x1x55x128xf32>

  //CHECK: [[SCALE:%.+]] = const.Declare
  //CHECK: [[TRANS:%.+]] = IE.Transpose
  //CHECK: IE.SDPAExtended([[ARG0]], [[ARG1]], [[TRANS]],  [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x1x55x128xf32>, tensor<1x1x80x128xf32>, tensor<1x1x128x80xf32>, tensor<1xf32> -> tensor<1x1x55x128xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Fuse_MM_SM_MM_WithConsecutiveTransposes
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x24x225x16xf16>, [[ARG1:%.+]]: tensor<1x24x225x16xf16>, [[ARG2:%.+]]: tensor<1x225x24x16xf16>)
func.func @Fuse_MM_SM_MM_WithConsecutiveTransposes(%arg0: tensor<1x24x225x16xf16>, %arg1: tensor<1x24x225x16xf16>, %arg2: tensor<1x225x24x16xf16>) -> tensor<1x24x225x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16> -> tensor<1x24x225x225xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x24x225x225xf16> -> tensor<1x24x225x225xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x225x24x16xf16> -> tensor<1x24x225x16xf16>
  %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x24x225x16xf16> -> tensor<1x24x16x225xf16>
  %4 = IE.MatMul(%1, %3) {transpose_b} : tensor<1x24x225x225xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
  return %4 : tensor<1x24x225x16xf16>

  // CHECK: [[COMPOSED_TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NHWC} : tensor<1x225x24x16xf16> -> tensor<1x24x16x225xf16>
  // CHECK: IE.SDPAExtended([[ARG0]], [[ARG1]], [[COMPOSED_TRANS]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
}

// -----

// CHECK-LABEL: @Fuse_MM_SM_MM_WithBatchToChannels
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<10x11x128x64xf16>, [[ARG1:%.+]]: tensor<10x11x128x64xf16>, [[ARG2:%.+]]: tensor<10x11x128x64xf16>)
func.func @Fuse_MM_SM_MM_WithBatchToChannels(%arg0: tensor<10x11x128x64xf16>, %arg1: tensor<10x11x128x64xf16>, %arg2: tensor<10x11x128x64xf16>) -> tensor<10x11x128x64xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16> -> tensor<10x11x128x128xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<10x11x128x128xf16> -> tensor<10x11x128x128xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<10x11x128x64xf16> -> tensor<10x11x64x128xf16>
  %3 = IE.MatMul(%1, %2) {transpose_b} : tensor<10x11x128x128xf16>, tensor<10x11x64x128xf16> -> tensor<10x11x128x64xf16>
  return %3 : tensor<10x11x128x64xf16>

  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]])
  // CHECK: [[RESHAPE_Q:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 110, 128, 64]} : tensor<10x11x128x64xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_K:%.+]] = IE.Reshape([[ARG1]]) {shape_value = [1, 110, 128, 64]} : tensor<10x11x128x64xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_V:%.+]] = IE.Reshape([[TRANS]]) {shape_value = [1, 110, 64, 128]} : tensor<10x11x64x128xf16> -> tensor<1x110x64x128xf16>
  // CHECK: [[SDPA:%.+]] = IE.SDPAExtended([[RESHAPE_Q]], [[RESHAPE_K]], [[RESHAPE_V]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>} : tensor<1x110x128x64xf16>, tensor<1x110x128x64xf16>, tensor<1x110x64x128xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_BACK:%.+]] = IE.Reshape([[SDPA]]) {shape_value = [10, 11, 128, 64]} : tensor<1x110x128x64xf16> -> tensor<10x11x128x64xf16>
  // CHECK: return [[RESHAPE_BACK]]
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

// CHECK-LABEL: @Fuse_SDPA_WithConsecutiveTransposes
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<28x64x128xf32>, [[ARG1:%.+]]: tensor<28x1024x128xf32>, [[ARG2:%.+]]: tensor<1024x28x128xf32>)
func.func @Fuse_SDPA_WithConsecutiveTransposes(%arg0: tensor<28x64x128xf32>, %arg1: tensor<28x1024x128xf32>, %arg2: tensor<1024x28x128xf32>) -> tensor<28x64x128xf32> {
  %cst = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
  %0 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2) -> (d1, d0, d2)>} : tensor<1024x28x128xf32> -> tensor<28x1024x128xf32>
  %1 = IE.SDPA(%arg0, %arg1, %0, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>} : tensor<28x64x128xf32>, tensor<28x1024x128xf32>, tensor<28x1024x128xf32>, tensor<1xf32> -> tensor<28x64x128xf32>
  return %1 : tensor<28x64x128xf32>

  // CHECK: [[SCALE:%.+]] = const.Declare
  // CHECK: [[COMPOSED_TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #HWC} : tensor<1024x28x128xf32> -> tensor<28x128x1024xf32>
  // CHECK: IE.SDPAExtended([[ARG0]], [[ARG1]], [[COMPOSED_TRANS]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<28x64x128xf32>, tensor<28x1024x128xf32>, tensor<28x128x1024xf32>, tensor<1xf32> -> tensor<28x64x128xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Fuse_SDPA_WithBatchToChannels
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<10x11x128x64xf16>, [[ARG1:%.+]]: tensor<10x11x128x64xf16>, [[ARG2:%.+]]: tensor<10x11x128x64xf16>)
func.func @Fuse_SDPA_WithBatchToChannels(%arg0: tensor<10x11x128x64xf16>, %arg1: tensor<10x11x128x64xf16>, %arg2: tensor<10x11x128x64xf16>) -> tensor<10x11x128x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<8.000000e+00> : tensor<1xf32>
  %1 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>} : tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<1xf32> -> tensor<10x11x128x64xf16>
  return %1 : tensor<10x11x128x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare
  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<10x11x128x64xf16> -> tensor<10x11x64x128xf16>
  // CHECK: [[RESHAPE_Q:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 110, 128, 64]} : tensor<10x11x128x64xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_K:%.+]] = IE.Reshape([[ARG1]]) {shape_value = [1, 110, 128, 64]} : tensor<10x11x128x64xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_V:%.+]] = IE.Reshape([[TRANS]]) {shape_value = [1, 110, 64, 128]} : tensor<10x11x64x128xf16> -> tensor<1x110x64x128xf16>
  // CHECK: [[SDPA:%.+]] = IE.SDPAExtended([[RESHAPE_Q]], [[RESHAPE_K]], [[RESHAPE_V]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x110x128x64xf16>, tensor<1x110x128x64xf16>, tensor<1x110x64x128xf16>, tensor<1xf32> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_BACK:%.+]] = IE.Reshape([[SDPA]]) {shape_value = [10, 11, 128, 64]} : tensor<1x110x128x64xf16> -> tensor<10x11x128x64xf16>
  // CHECK: return [[RESHAPE_BACK]]
}

// -----

// CHECK-LABEL: @Fuse_SDPA_WithIllegalBatchToChannels
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<10x11x128x64xf16>, [[ARG1:%.+]]: tensor<10x11x128x64xf16>, [[ARG2:%.+]]: tensor<10x11x128x64xf16>, [[ARG3:%.+]]: tensor<10x1x128x64xf16>)
func.func @Fuse_SDPA_WithIllegalBatchToChannels(%arg0: tensor<10x11x128x64xf16>, %arg1: tensor<10x11x128x64xf16>, %arg2: tensor<10x11x128x64xf16>, %arg3: tensor<10x1x128x64xf16>) -> tensor<10x11x128x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<8.000000e+00> : tensor<1xf32>
  %1 = IE.SDPA(%arg0, %arg1, %arg2, %arg3, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<10x1x128x64xf16>, tensor<1xf32> -> tensor<10x11x128x64xf16>
  return %1 : tensor<10x11x128x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare
  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<10x11x128x64xf16> -> tensor<10x11x64x128xf16>
  // CHECK: [[SDPA:%.+]] = IE.SDPAExtended([[ARG0]], [[ARG1]], [[TRANS]], [[ARG3]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<10x11x64x128xf16>, tensor<10x1x128x64xf16>, tensor<1xf32> -> tensor<10x11x128x64xf16>
  // CHECK: return [[SDPA]]
}

// -----

// CHECK-LABEL: @Fuse_SDPA_With3DMaskNotBroadcastable
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<64x3x128x64xf16>, [[ARG1:%.+]]: tensor<64x3x128x64xf16>, [[ARG2:%.+]]: tensor<64x3x128x64xf16>, [[ARG3:%.+]]: tensor<3x128x128xf16>)
func.func @Fuse_SDPA_With3DMaskNotBroadcastable(%arg0: tensor<64x3x128x64xf16>, %arg1: tensor<64x3x128x64xf16>, %arg2: tensor<64x3x128x64xf16>, %arg3: tensor<3x128x128xf16>) -> tensor<64x3x128x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<8.000000e+00> : tensor<1xf32>
  %1 = IE.SDPA(%arg0, %arg1, %arg2, %arg3, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : tensor<64x3x128x64xf16>, tensor<64x3x128x64xf16>, tensor<64x3x128x64xf16>, tensor<3x128x128xf16>, tensor<1xf32> -> tensor<64x3x128x64xf16>
  return %1 : tensor<64x3x128x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare
  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<64x3x128x64xf16> -> tensor<64x3x64x128xf16>
  // CHECK: [[SDPA:%.+]] = IE.SDPAExtended([[ARG0]], [[ARG1]], [[TRANS]], [[ARG3]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<64x3x128x64xf16>, tensor<64x3x128x64xf16>, tensor<64x3x64x128xf16>, tensor<3x128x128xf16>, tensor<1xf32> -> tensor<64x3x128x64xf16>
  // CHECK: return [[SDPA]]
}

// -----

// CHECK-LABEL: @NotFuse_MM_SM_MM_With5DMask
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x24x225x16xf16>, [[ARG1:%.+]]: tensor<1x24x225x16xf16>, [[ARG2:%.+]]: tensor<1x24x225x16xf16>, [[ARG3:%.+]]: tensor<1x1x1x225x225xf16>)
func.func @NotFuse_MM_SM_MM_With5DMask(%arg0: tensor<1x24x225x16xf16>, %arg1: tensor<1x24x225x16xf16>, %arg2: tensor<1x24x225x16xf16>, %arg3: tensor<1x1x1x225x225xf16>) -> tensor<1x24x225x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16> -> tensor<1x24x225x225xf16>
  %1 = IE.Reshape(%0) {shape_value = [1, 1, 24, 225, 225]} : tensor<1x24x225x225xf16> -> tensor<1x1x24x225x225xf16>
  %2 = IE.Add(%1, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x24x225x225xf16>, tensor<1x1x1x225x225xf16> -> tensor<1x1x24x225x225xf16>
  %3 = IE.Reshape(%2) {shape_value = [1, 24, 225, 225]} : tensor<1x1x24x225x225xf16> -> tensor<1x24x225x225xf16>
  %4 = IE.SoftMax(%3) {axisInd = 3 : i64} : tensor<1x24x225x225xf16> -> tensor<1x24x225x225xf16>
  %5 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x24x225x16xf16> -> tensor<1x24x16x225xf16>
  %6 = IE.MatMul(%4, %5) {transpose_b} : tensor<1x24x225x225xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
  return %6 : tensor<1x24x225x16xf16>

  // CHECK-NOT: IE.SDPAExtended
  // CHECK: IE.MatMul
}

// -----

// CHECK-LABEL: @NoFuseSDPA_ReshapeSwapsLastTwoDimsBetweenMatMulAndSoftmax
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x4x320x32xf16>, [[ARG1:%.+]]: tensor<1x4x1x32xf16>, [[ARG2:%.+]]: tensor<1x4x32x320xf16>)
func.func @NoFuseSDPA_ReshapeSwapsLastTwoDimsBetweenMatMulAndSoftmax(%arg0: tensor<1x4x320x32xf16>, %arg1: tensor<1x4x1x32xf16>, %arg2: tensor<1x4x32x320xf16>) -> tensor<1x4x1x32xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x4x320x32xf16>, tensor<1x4x1x32xf16> -> tensor<1x4x320x1xf16>
  %1 = IE.Reshape(%0) {shape_value = [1, 4, 1, 320]} : tensor<1x4x320x1xf16> -> tensor<1x4x1x320xf16>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x4x1x320xf16> -> tensor<1x4x1x320xf16>
  %3 = IE.MatMul(%2, %arg2) {transpose_b} : tensor<1x4x1x320xf16>, tensor<1x4x32x320xf16> -> tensor<1x4x1x32xf16>
  return %3 : tensor<1x4x1x32xf16>

  // CHECK-NOT: IE.SDPAExtended
}

// -----

// CHECK-LABEL: @NoFuseSDPA_TransposeSwapsLastTwoDimsBetweenMatMulAndSoftmax
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x4x320x32xf16>, [[ARG1:%.+]]: tensor<1x4x320x32xf16>, [[ARG2:%.+]]: tensor<1x4x32x320xf16>)
func.func @NoFuseSDPA_TransposeSwapsLastTwoDimsBetweenMatMulAndSoftmax(%arg0: tensor<1x4x320x32xf16>, %arg1: tensor<1x4x320x32xf16>, %arg2: tensor<1x4x32x320xf16>) -> tensor<1x4x320x32xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x4x320x32xf16>, tensor<1x4x320x32xf16> -> tensor<1x4x320x320xf16>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x4x320x320xf16> -> tensor<1x4x320x320xf16>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x4x320x320xf16> -> tensor<1x4x320x320xf16>
  %3 = IE.MatMul(%2, %arg2) {transpose_b} : tensor<1x4x320x320xf16>, tensor<1x4x32x320xf16> -> tensor<1x4x320x32xf16>
  return %3 : tensor<1x4x320x32xf16>

  // CHECK-NOT: IE.SDPAExtended
}
