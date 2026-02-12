//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-sdpa-pattern %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @UnrollSDPAPatternWithMatmulsHaveDifferentShrinkingBehavior
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x128x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[MASK:%.+]]: tensor<1x1x128x128xf32>, [[V_IN:%.+]]: tensor<1x1x1x128x128xf32>)
func.func @UnrollSDPAPatternWithMatmulsHaveDifferentShrinkingBehavior(%arg0: tensor<1x8x128x128xf32>, %arg1: tensor<1x8x128x128xf32>, %mask: tensor<1x1x128x128xf32>, %v_in: tensor<1x1x1x128x128xf32>) -> tensor<1x8x128x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[1, 1, 8, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x128x128xf32>, tensor<5xsi64> -> tensor<1x1x8x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 128, 128]} : tensor<1x1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<1x8x128x128xf32>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %3 = IE.MatMul(%2, %transpose) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  return %3 : tensor<1x8x128x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  
  // CHECK: [[SLICE_Q_0:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_2:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_3:%.+]] = IE.Slice [[ARG0]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_4:%.+]] = IE.Slice [[ARG0]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_5:%.+]] = IE.Slice [[ARG0]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_6:%.+]] = IE.Slice [[ARG0]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_7:%.+]] = IE.Slice [[ARG0]] [0, 7, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_0:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_1:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_2:%.+]] = IE.Slice [[ARG1]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_3:%.+]] = IE.Slice [[ARG1]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_4:%.+]] = IE.Slice [[ARG1]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_5:%.+]] = IE.Slice [[ARG1]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_6:%.+]] = IE.Slice [[ARG1]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_7:%.+]] = IE.Slice [[ARG1]] [0, 7, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_0:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_2:%.+]] = IE.Slice [[TRANSPOSE]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_3:%.+]] = IE.Slice [[TRANSPOSE]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_4:%.+]] = IE.Slice [[TRANSPOSE]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_5:%.+]] = IE.Slice [[TRANSPOSE]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_6:%.+]] = IE.Slice [[TRANSPOSE]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_7:%.+]] = IE.Slice [[TRANSPOSE]] [0, 7, 0, 0] [1, 1, 128, 128]

  // CHECK: [[MATMUL_1_0:%.+]] = IE.MatMul([[SLICE_Q_0]], [[SLICE_K_0]]) {transpose_b}
  // CHECK: [[ADD_0:%.+]] = IE.Add([[MATMUL_1_0]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_0:%.+]] = IE.SoftMax([[ADD_0]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_0:%.+]] = IE.MatMul([[SOFTMAX_0]], [[SLICE_V_0]]) {transpose_b}
  
  // CHECK: [[MATMUL_1_1:%.+]] = IE.MatMul([[SLICE_Q_1]], [[SLICE_K_1]]) {transpose_b}
  // CHECK: [[ADD_1:%.+]] = IE.Add([[MATMUL_1_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_1:%.+]] = IE.SoftMax([[ADD_1]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_1:%.+]] = IE.MatMul([[SOFTMAX_1]], [[SLICE_V_1]]) {transpose_b}
  
  // CHECK: [[MATMUL_1_2:%.+]] = IE.MatMul([[SLICE_Q_2]], [[SLICE_K_2]]) {transpose_b}
  // CHECK: [[ADD_2:%.+]] = IE.Add([[MATMUL_1_2]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_2:%.+]] = IE.SoftMax([[ADD_2]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_2:%.+]] = IE.MatMul([[SOFTMAX_2]], [[SLICE_V_2]]) {transpose_b}
  
  // CHECK: [[MATMUL_1_3:%.+]] = IE.MatMul([[SLICE_Q_3]], [[SLICE_K_3]]) {transpose_b}
  // CHECK: [[ADD_3:%.+]] = IE.Add([[MATMUL_1_3]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_3:%.+]] = IE.SoftMax([[ADD_3]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_3:%.+]] = IE.MatMul([[SOFTMAX_3]], [[SLICE_V_3]]) {transpose_b}
  
  // CHECK: [[MATMUL_1_4:%.+]] = IE.MatMul([[SLICE_Q_4]], [[SLICE_K_4]]) {transpose_b}
  // CHECK: [[ADD_4:%.+]] = IE.Add([[MATMUL_1_4]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_4:%.+]] = IE.SoftMax([[ADD_4]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_4:%.+]] = IE.MatMul([[SOFTMAX_4]], [[SLICE_V_4]]) {transpose_b}
  
  // CHECK: [[MATMUL_1_5:%.+]] = IE.MatMul([[SLICE_Q_5]], [[SLICE_K_5]]) {transpose_b}
  // CHECK: [[ADD_5:%.+]] = IE.Add([[MATMUL_1_5]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_5:%.+]] = IE.SoftMax([[ADD_5]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_5:%.+]] = IE.MatMul([[SOFTMAX_5]], [[SLICE_V_5]]) {transpose_b}
  
  // CHECK: [[MATMUL_1_6:%.+]] = IE.MatMul([[SLICE_Q_6]], [[SLICE_K_6]]) {transpose_b}
  // CHECK: [[ADD_6:%.+]] = IE.Add([[MATMUL_1_6]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_6:%.+]] = IE.SoftMax([[ADD_6]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_6:%.+]] = IE.MatMul([[SOFTMAX_6]], [[SLICE_V_6]]) {transpose_b}
  
  // CHECK: [[MATMUL_1_7:%.+]] = IE.MatMul([[SLICE_Q_7]], [[SLICE_K_7]]) {transpose_b}
  // CHECK: [[ADD_7:%.+]] = IE.Add([[MATMUL_1_7]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_7:%.+]] = IE.SoftMax([[ADD_7]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_7:%.+]] = IE.MatMul([[SOFTMAX_7]], [[SLICE_V_7]]) {transpose_b}
  
  // CHECK: [[CONCAT:%.+]] = IE.Concat([[MATMUL_V_0]], [[MATMUL_V_1]], [[MATMUL_V_2]], [[MATMUL_V_3]], [[MATMUL_V_4]], [[MATMUL_V_5]], [[MATMUL_V_6]], [[MATMUL_V_7]])
  // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @DontUnrollSDPAPatternWithMatmulsHaveSameShrinkingBehavior
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x128x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[MASK:%.+]]: tensor<1x1x128x128xf32>, [[ARG2:%.+]]: tensor<1x8x128x128xf32>)
func.func @DontUnrollSDPAPatternWithMatmulsHaveSameShrinkingBehavior(%arg0: tensor<1x8x128x128xf32>, %arg1: tensor<1x8x128x128xf32>, %mask: tensor<1x1x128x128xf32>, %arg2: tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<1x8x128x128xf32>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %3 = IE.MatMul(%2, %arg2) : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  return %3 : tensor<1x8x128x128xf32>

  // CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK: [[ADD:%.+]] = IE.Add([[MATMUL_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]])
  // CHECK: return [[MATMUL_V]]
}

// -----
	
// CHECK-LABEL: @DontUnrollSDPAPatternBatch1
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<8x1x128x128xf32>, [[ARG1:%.+]]: tensor<8x1x128x128xf32>, [[MASK:%.+]]: tensor<1x1x128x128xf32>, [[V_IN:%.+]]: tensor<8x1x1x128x128xf32>)
func.func @DontUnrollSDPAPatternBatch1(%arg0: tensor<8x1x128x128xf32>, %arg1: tensor<8x1x128x128xf32>, %mask: tensor<1x1x128x128xf32>, %v_in: tensor<8x1x1x128x128xf32>) -> tensor<8x1x128x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[8, 1, 1, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<8x1x1x128x128xf32>, tensor<5xsi64> -> tensor<8x1x1x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [8, 1, 128, 128]} : tensor<8x1x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<8x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<8x1x128x128xf32>, tensor<8x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x1x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<8x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %3 = IE.MatMul(%2, %transpose) {transpose_b} : tensor<8x1x128x128xf32>, tensor<8x1x128x128xf32> -> tensor<8x1x128x128xf32>
  return %3 : tensor<8x1x128x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

	// CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
	// CHECK: [[ADD:%.+]] = IE.Add([[MATMUL_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
	// CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 3 : i64}
	// CHECK: [[MATMUL_V:%.+]] = IE.MatMul([[SOFTMAX]], [[TRANSPOSE]])
	// CHECK: return [[MATMUL_V]]
}

// -----

// CHECK-LABEL: @DontUnrollSDPAPatternNoAdd
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x128x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[V_IN:%.+]]: tensor<1x1x1x128x128xf32>)
func.func @DontUnrollSDPAPatternNoAdd(%arg0: tensor<1x8x128x128xf32>, %arg1: tensor<1x8x128x128xf32>, %v_in: tensor<1x1x1x128x128xf32>) -> tensor<1x8x128x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[1, 1, 8, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x128x128xf32>, tensor<5xsi64> -> tensor<1x1x8x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 128, 128]} : tensor<1x1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %2 = IE.MatMul(%1, %transpose) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  return %2 : tensor<1x8x128x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

  // CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
	// CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MATMUL_1]]) {axisInd = 3 : i64}
	// CHECK: [[MATMUL_V:%.+]] = IE.MatMul([[SOFTMAX]], [[TRANSPOSE]]) {transpose_b}
  // CHECK: return [[MATMUL_V]]
}
