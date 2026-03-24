//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-sdpa-extended %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @DecomposeBasicSDPAExtended
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeBasicSDPAExtended(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeSDPAExtendedWithMask
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x8x64x64xf16>, [[ARG4:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeSDPAExtendedWithMask(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x8x64x64xf16>, %arg4: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x8x64x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[MASKED:%.+]] = IE.Add([[QK]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeSDPAExtendedWithCustomScale
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1xf32>)
func.func @DecomposeSDPAExtendedWithCustomScale(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeSDPAExtendedWithBias
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>, [[ARG4:%.+]]: tensor<1x8x64x64xf16>)
func.func @DecomposeSDPAExtendedWithBias(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x1x1x1xf32>, %arg4: tensor<1x8x64x64xf16>) -> tensor<1x8x64x32xf16> {
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 1>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x1x1x1xf32>, tensor<1x8x64x64xf16> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[BIASED:%.+]] = IE.Add([[QK]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[BIASED]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeSDPAExtendedWithMaskAndScale
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x8x64x64xf16>, [[ARG4:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeSDPAExtendedWithMaskAndScale(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x8x64x64xf16>, %arg4: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x8x64x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[MASKED:%.+]] = IE.Add([[QK]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @NotDecomposeLegalSDPAExtended
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x192x225x16xf16>, [[ARG1:%.+]]: tensor<1x192x225x16xf16>, [[ARG2:%.+]]: tensor<1x192x16x225xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @NotDecomposeLegalSDPAExtended(%arg0: tensor<1x192x225x16xf16>, %arg1: tensor<1x192x225x16xf16>, %arg2: tensor<1x192x16x225xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x192x225x16xf16> {
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x192x225x16xf16>, tensor<1x192x225x16xf16>, tensor<1x192x16x225xf16>, tensor<1x1x1x1xf32> -> tensor<1x192x225x16xf16>
  return %0 : tensor<1x192x225x16xf16>

  // CHECK: [[SDPA:%.+]] = IE.SDPAExtended([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]])
  // CHECK: return [[SDPA]]
}

// -----

// CHECK-LABEL: @DecomposeSDPAExtendedRank3
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<8x64x32xf16>, [[ARG1:%.+]]: tensor<8x64x32xf16>, [[ARG2:%.+]]: tensor<8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1xf32>)
func.func @DecomposeSDPAExtendedRank3(%arg0: tensor<8x64x32xf16>, %arg1: tensor<8x64x32xf16>, %arg2: tensor<8x32x64xf16>, %arg3: tensor<1x1x1xf32>) -> tensor<8x64x32xf16> {
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<8x64x32xf16>, tensor<8x64x32xf16>, tensor<8x32x64xf16>, tensor<1x1x1xf32> -> tensor<8x64x32xf16>
  return %0 : tensor<8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 2 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeSDPAExtendedWithAllZeroMask
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeSDPAExtendedWithAllZeroMask(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %mask = const.Declare tensor<1x8x64x64xf16> = dense<0.0> : tensor<1x8x64x64xf16>
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %mask, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x8x64x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK-DAG: [[MASK:%.+]] = const.Declare tensor<1x8x64x64xf16> = dense<0.000000e+00>
  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK-NOT: IE.Add
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeSDPAExtendedWithAllOneScale
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>)
func.func @DecomposeSDPAExtendedWithAllOneScale(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>) -> tensor<1x8x64x32xf16> {
  %scale = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
  %0 = IE.SDPAExtended(%arg0, %arg1, %arg2, %scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00>
  // CHECK: [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK-NOT: IE.Multiply
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeSDPAExtendedWithRecursiveVPreprocessing
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<28x64x128xf32>, [[ARG1:%.+]]: tensor<28x1024x128xf32>, [[ARG2:%.+]]: tensor<1024x1x3584xf32>, [[ARG3:%.+]]: tensor<3584x3584xf32>, [[ARG4:%.+]]: tensor<28x1x1024xf32>)
func.func @DecomposeSDPAExtendedWithRecursiveVPreprocessing(%arg0: tensor<28x64x128xf32>, %arg1: tensor<28x1024x128xf32>, %arg2: tensor<1024x1x3584xf32>, %arg3: tensor<3584x3584xf32>, %arg4: tensor<28x1x1024xf32>) -> tensor<28x64x128xf32> {
  %cst_bias = const.Declare tensor<1x1x3584xf32> = dense<1.0> : tensor<1x1x3584xf32>
  %cst_scale = const.Declare tensor<1xf32> = dense<0.088388346> : tensor<1xf32>

  // V preprocessing chain: MatMul -> Add -> AffineReshape -> Transpose
  %0 = IE.MatMul(%arg2, %arg3) {transpose_b} : tensor<1024x1x3584xf32>, tensor<3584x3584xf32> -> tensor<1024x1x3584xf32>
  %1 = IE.Add(%0, %cst_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1x3584xf32>, tensor<1x1x3584xf32> -> tensor<1024x1x3584xf32>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1, 2]], shape_value = [1024, 28, 128]} : tensor<1024x1x3584xf32> -> tensor<1024x28x128xf32>
  %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : tensor<1024x28x128xf32> -> tensor<28x128x1024xf32>

  %4 = IE.SDPAExtended(%arg0, %arg1, %3, %arg4, %cst_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<28x64x128xf32>, tensor<28x1024x128xf32>, tensor<28x128x1024xf32>, tensor<28x1x1024xf32>, tensor<1xf32> -> tensor<28x64x128xf32>
  return %4 : tensor<28x64x128xf32>

  // CHECK: [[CST_BIAS:%.+]] = const.Declare tensor<1x1x3584xf32> = dense<1.000000e+00>
  // CHECK: [[CST_SCALE:%.+]] = const.Declare tensor<1xf32> = dense<{{.*}}>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[CST_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[MASKED:%.+]] = IE.Add([[QK]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 2 : i64}

  // V preprocessing moved after softmax - the operations are cloned
  // CHECK: [[V_MATMUL:%.+]] = IE.MatMul([[ARG2]], [[ARG3]]) {transpose_b}
  // CHECK: [[V_ADD:%.+]] = IE.Add([[V_MATMUL]], [[CST_BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[V_RESHAPE:%.+]] = IE.AffineReshape([[V_ADD]]) {dim_mapping = {{\[\[}}0], [0], [1, 2]], shape_value = [1024, 28, 128]}
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_RESHAPE]])

  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[V_TRANSPOSE]]) {transpose_b}
  // CHECK: return [[OUT]]
}
