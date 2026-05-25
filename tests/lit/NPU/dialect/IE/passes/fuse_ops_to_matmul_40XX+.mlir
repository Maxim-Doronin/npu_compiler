//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-ops-to-matmul="enable-grouped-matmul=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumToMatMul
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<6x256x1x8192xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<6x1x256x8192xf16>
func.func @ConvertBroadcastMultiplyReduceSumToMatMul(%arg0: tensor<6x256x1x8192xf16>, %arg1: tensor<6x1x256x8192xf16>) -> tensor<6x256x256x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x256x1x8192xf16>, tensor<6x1x256x8192xf16> -> tensor<6x256x256x8192xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<6x256x256x8192xf16> -> tensor<6x256x256x1xf16>
  return %red : tensor<6x256x256x1xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b} : tensor<6x256x8192xf16>, tensor<6x256x8192xf16> -> tensor<6x256x256xf16>
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [6, 256, 256, 1]}
  // CHECK:       return [[OUT]] : tensor<6x256x256x1xf16>
}

// -----

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumToMatMulSingleAxis
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<6x256x1x128xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<6x1x64x128xf16>
func.func @ConvertBroadcastMultiplyReduceSumToMatMulSingleAxis(%arg0: tensor<6x256x1x128xf16>, %arg1: tensor<6x1x64x128xf16>) -> tensor<6x256x64x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x256x1x128xf16>, tensor<6x1x64x128xf16> -> tensor<6x256x64x128xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<6x256x64x128xf16> -> tensor<6x256x64x1xf16>
  return %red : tensor<6x256x64x1xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [6, 256, 128]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [6, 64, 128]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [6, 256, 64, 1]}
  // CHECK:       return [[OUT]]
}

// -----

// CHECK-LABEL: @NotConvertSingleAxisBroadcastMultiplyReduceSum
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x8192xf16>
func.func @NotConvertSingleAxisBroadcastMultiplyReduceSum(%arg0: tensor<1x256x8192xf16>, %arg1: tensor<1x1x8192xf16>) -> tensor<1x256x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x8192xf16>, tensor<1x1x8192xf16> -> tensor<1x256x8192xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [2], keep_dims} : tensor<1x256x8192xf16> -> tensor<1x256x1xf16>
  return %red : tensor<1x256x1xf16>

  // CHECK-NOT:   IE.MatMul
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT]], %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[REDUCESUM:%.+]] = IE.ReduceSum([[MULTIPLY]]) {axes_value = [2], keep_dims}
  // CHECK:       return [[REDUCESUM]] : tensor<1x256x1xf16>
}

// -----

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumToMatMulKeepDimsFalse
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<6x256x1x8192xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<6x1x256x8192xf16>
func.func @ConvertBroadcastMultiplyReduceSumToMatMulKeepDimsFalse(%arg0: tensor<6x256x1x8192xf16>, %arg1: tensor<6x1x256x8192xf16>) -> tensor<6x256x256xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x256x1x8192xf16>, tensor<6x1x256x8192xf16> -> tensor<6x256x256x8192xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3]} : tensor<6x256x256x8192xf16> -> tensor<6x256x256xf16>
  return %red : tensor<6x256x256xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [6, 256, 256]}
  // CHECK:       return [[OUT]] : tensor<6x256x256xf16>
}

// -----

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumToMatMulRank6
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<1x4x256x1x64x128xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<1x4x1x256x64x128xf16>
func.func @ConvertBroadcastMultiplyReduceSumToMatMulRank6(%arg0: tensor<1x4x256x1x64x128xf16>, %arg1: tensor<1x4x1x256x64x128xf16>) -> tensor<1x4x256x256x64x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x1x64x128xf16>, tensor<1x4x1x256x64x128xf16> -> tensor<1x4x256x256x64x128xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [5], keep_dims} : tensor<1x4x256x256x64x128xf16> -> tensor<1x4x256x256x64x1xf16>
  return %red : tensor<1x4x256x256x64x1xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS_T:%.+]] = IE.Transpose([[INPUT_A]]) {{.+}} : tensor<1x4x256x1x64x128xf16> -> tensor<1x4x64x256x1x128xf16>
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[LHS_T]]) {shape_value = [256, 256, 128]}
  // CHECK:       [[RHS_T:%.+]] = IE.Transpose([[INPUT_B]]) {{.+}} : tensor<1x4x1x256x64x128xf16> -> tensor<1x4x64x256x1x128xf16>
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[RHS_T]]) {shape_value = [256, 256, 128]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[EXP:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 4, 64, 256, 256]}
  // CHECK:       [[INV:%.+]] = IE.Transpose([[EXP]]) {{.+}} : tensor<1x4x64x256x256xf16> -> tensor<1x4x256x256x64xf16>
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[INV]]) {shape_value = [1, 4, 256, 256, 64, 1]}
  // CHECK:       return [[OUT]] : tensor<1x4x256x256x64x1xf16>
}

// -----

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumNonLastAxis
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<1x4x64x256x128x1xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<1x4x64x256x1x64xf16>
func.func @ConvertBroadcastMultiplyReduceSumNonLastAxis(%arg0: tensor<1x4x64x256x128x1xf16>, %arg1: tensor<1x4x64x256x1x64xf16>) -> tensor<1x4x64x1x128x64xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x64x256x128x1xf16>, tensor<1x4x64x256x1x64xf16> -> tensor<1x4x64x256x128x64xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<1x4x64x256x128x64xf16> -> tensor<1x4x64x1x128x64xf16>
  return %red : tensor<1x4x64x1x128x64xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS_K:%.+]] = IE.Transpose([[INPUT_A]]) {{.+}} : tensor<1x4x64x256x128x1xf16> -> tensor<1x4x64x128x1x256xf16>
  // CHECK:       [[RHS_K:%.+]] = IE.Transpose([[INPUT_B]]) {{.+}} : tensor<1x4x64x256x1x64xf16> -> tensor<1x4x64x1x64x256xf16>
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[LHS_K]]) {shape_value = [256, 128, 256]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[RHS_K]]) {shape_value = [256, 64, 256]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 4, 64, 1, 128, 64]}
  // CHECK:       return [[OUT]] : tensor<1x4x64x1x128x64xf16>
}

// -----

// CHECK-LABEL: @NotConvertMultiplyWithMultipleUses
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<4x256x1x128xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<4x1x256x128xf16>
func.func @NotConvertMultiplyWithMultipleUses(%arg0: tensor<4x256x1x128xf16>, %arg1: tensor<4x1x256x128xf16>) -> (tensor<4x256x256x1xf16>, tensor<4x256x256x128xf16>) {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x256x1x128xf16>, tensor<4x1x256x128xf16> -> tensor<4x256x256x128xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<4x256x256x128xf16> -> tensor<4x256x256x1xf16>
  return %red, %mul : tensor<4x256x256x1xf16>, tensor<4x256x256x128xf16>

  // CHECK-NOT:   IE.MatMul
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_A]], [[INPUT_B]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[REDUCESUM:%.+]] = IE.ReduceSum([[MULTIPLY]]) {axes_value = [3], keep_dims}
  // CHECK:       return [[REDUCESUM]], [[MULTIPLY]]
}
