//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-sdpa-extended --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX||arch-NPU50XX



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

// CHECK-LABEL: @NotFuse_MM_SM_MM_Batch
func.func @NotFuse_MM_SM_MM_Batch(%arg0: tensor<1x11x225x16xf16>, %arg1: tensor<1x11x225x16xf16>, %arg2: tensor<1x11x225x16xf16>) -> tensor<1x11x225x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x11x225x16xf16>, tensor<1x11x225x16xf16> -> tensor<1x11x225x225xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x11x225x225xf16> -> tensor<1x11x225x225xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x11x225x16xf16> -> tensor<1x11x16x225xf16>
  %3 = IE.MatMul(%1, %2) {transpose_b} : tensor<1x11x225x225xf16>, tensor<1x11x16x225xf16> -> tensor<1x11x225x16xf16>
  return %3 : tensor<1x11x225x16xf16>

  // CHECK-NOT: IE.SDPAExtended
}

// -----

// CHECK-LABEL: @NotFuseMM_SM_MM_Height
func.func @NotFuseMM_SM_MM_Height(%arg0: tensor<1x11x225x16xf16>, %arg1: tensor<1x11x225x16xf16>, %arg2: tensor<1x11x225x16xf16>) -> tensor<1x11x225x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x11x225x16xf16>, tensor<1x11x225x16xf16> -> tensor<1x11x225x225xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x11x225x225xf16> -> tensor<1x11x225x225xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x11x225x16xf16> -> tensor<1x11x16x225xf16>
  %3 = IE.MatMul(%1, %2) {transpose_b} : tensor<1x11x225x225xf16>, tensor<1x11x16x225xf16> -> tensor<1x11x225x16xf16>
  return %3 : tensor<1x11x225x16xf16>

  // CHECK-NOT: IE.SDPAExtended
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
