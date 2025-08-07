//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-multi-zp-quantization-pattern %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @DecomposeWACPatternWithPostFP32MatMul
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf32>)
func.func @DecomposeWACPatternWithPostFP32MatMul(%act : tensor<1x3x4xf32>) -> tensor<1x3x4xf32> {
  %weights = const.Declare tensor<4x2x2xf16> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xui8>, [#const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %1 = IE.Subtract(%weights, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.MatMul(%5, %4) {transpose_b} : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %7 : tensor<1x3x4xf32>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK-DAG: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x2xf16> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xui8>, [#const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  // CHECK:     [[MULTIPLY_1:%.+]] = IE.Multiply([[WEIGHTS]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK:     [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY_1]])
  // CHECK:     [[CONVERT_2:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK:     [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK:     [[MATMUL_1:%.+]] = IE.MatMul([[RESHAPE_2]], [[CONVERT_2]]) {transpose_b} : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK:     [[RESHAPE_3:%.+]] = IE.Reshape([[RESHAPE_2]]) {shape_value = [3, 2, 2]} : tensor<3x4xf32> -> tensor<3x2x2xf32>
  // CHECK:     [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE_3]]) {axes_value = [2]} : tensor<3x2x2xf32> -> tensor<3x2xf32>
  // CHECK:     [[MULTIPLY_2:%.+]] = IE.Multiply([[SCALE]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x1xf16>
  // CHECK:     [[RESHAPE_4:%.+]] = IE.Reshape([[MULTIPLY_2]]) {shape_value = [4, 2]} : tensor<4x2x1xf16> -> tensor<4x2xf16>
  // CHECK:     [[CONVERT_3:%.+]]  = IE.Convert([[RESHAPE_4]]) {dstElemType = f32} : tensor<4x2xf16> -> tensor<4x2xf32>
  // CHECK:     [[MATMUL_2:%.+]]  = IE.MatMul([[REDUCE_SUM]], [[CONVERT_3]]) {transpose_b} : tensor<3x2xf32>, tensor<4x2xf32> -> tensor<3x4xf32>
  // CHECK:     [[SUBTRACT:%.+]]  = IE.Subtract([[MATMUL_1]], [[MATMUL_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x4xf32>, tensor<3x4xf32> -> tensor<3x4xf32>
  // CHECK:     [[RES:%.+]]  = IE.Reshape([[SUBTRACT]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK:     return [[RES]]
}

// -----

// CHECK-LABEL: @DecomposeWACPatternWithPostFP32FC
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf32>)
func.func @DecomposeWACPatternWithPostFP32FC(%act : tensor<1x3x4xf32>) -> tensor<1x3x4xf32> {
  %weights = const.Declare tensor<4x2x2xf16> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xui8>, [#const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %1 = IE.Subtract(%weights, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.FullyConnected(%5, %4) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %7 : tensor<1x3x4xf32>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK-DAG: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x2xf16> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xui8>, [#const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  // CHECK:     [[MULTIPLY_1:%.+]] = IE.Multiply([[WEIGHTS]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK:     [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY_1]])
  // CHECK:     [[CONVERT_2:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK:     [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK:     [[FC_1:%.+]] = IE.FullyConnected([[RESHAPE_2]], [[CONVERT_2]]) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK:     [[RESHAPE_3:%.+]] = IE.Reshape([[RESHAPE_2]]) {shape_value = [3, 2, 2]} : tensor<3x4xf32> -> tensor<3x2x2xf32>
  // CHECK:     [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE_3]]) {axes_value = [2]} : tensor<3x2x2xf32> -> tensor<3x2xf32>
  // CHECK:     [[MULTIPLY_2:%.+]] = IE.Multiply([[SCALE]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x1xf16>
  // CHECK:     [[RESHAPE_4:%.+]] = IE.Reshape([[MULTIPLY_2]]) {shape_value = [4, 2]} : tensor<4x2x1xf16> -> tensor<4x2xf16>
  // CHECK:     [[CONVERT_3:%.+]]  = IE.Convert([[RESHAPE_4]]) {dstElemType = f32} : tensor<4x2xf16> -> tensor<4x2xf32>
  // CHECK:     [[FC_2:%.+]]  = IE.FullyConnected([[REDUCE_SUM]], [[CONVERT_3]]) : tensor<3x2xf32>, tensor<4x2xf32> -> tensor<3x4xf32>
  // CHECK:     [[SUBTRACT:%.+]]  = IE.Subtract([[FC_1]], [[FC_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x4xf32>, tensor<3x4xf32> -> tensor<3x4xf32>
  // CHECK:     [[RES:%.+]]  = IE.Reshape([[SUBTRACT]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK:     return [[RES]]
}

// -----

// CHECK-LABEL: @DecomposeWACPatternWithPostFP16MatMul
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf16>)
func.func @DecomposeWACPatternWithPostFP16MatMul(%act : tensor<1x3x4xf16>) -> tensor<1x3x4xf16> {
  %weights = const.Declare tensor<4x2x2xf16> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xui8>, [#const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %1 = IE.Subtract(%weights, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf16> -> tensor<3x4xf16>
  %6 = IE.MatMul(%5, %3) {transpose_b} : tensor<3x4xf16>, tensor<4x4xf16> -> tensor<3x4xf16>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf16> -> tensor<1x3x4xf16>
  return %7 : tensor<1x3x4xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK-DAG: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x2xf16> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xui8>, [#const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  // CHECK:     [[MULTIPLY_1:%.+]] = IE.Multiply([[WEIGHTS]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK:     [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY_1]])
  // CHECK:     [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf16> -> tensor<3x4xf16>
  // CHECK:     [[MATMUL_1:%.+]] = IE.MatMul([[RESHAPE_2]], [[RESHAPE_1]]) {transpose_b} : tensor<3x4xf16>, tensor<4x4xf16> -> tensor<3x4xf16>
  // CHECK:     [[RESHAPE_3:%.+]] = IE.Reshape([[RESHAPE_2]]) {shape_value = [3, 2, 2]} : tensor<3x4xf16> -> tensor<3x2x2xf16>
  // CHECK:     [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE_3]]) {axes_value = [2]} : tensor<3x2x2xf16> -> tensor<3x2xf16>
  // CHECK:     [[MULTIPLY_2:%.+]] = IE.Multiply([[SCALE]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x1xf16>
  // CHECK:     [[RESHAPE_4:%.+]] = IE.Reshape([[MULTIPLY_2]]) {shape_value = [4, 2]} : tensor<4x2x1xf16> -> tensor<4x2xf16>
  // CHECK:     [[MATMUL_2:%.+]]  = IE.MatMul([[REDUCE_SUM]], [[RESHAPE_4]]) {transpose_b} : tensor<3x2xf16>, tensor<4x2xf16> -> tensor<3x4xf16>
  // CHECK:     [[SUBTRACT:%.+]]  = IE.Subtract([[MATMUL_1]], [[MATMUL_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x4xf16>, tensor<3x4xf16> -> tensor<3x4xf16>
  // CHECK:     [[RES:%.+]]  = IE.Reshape([[SUBTRACT]]) {shape_value = [1, 3, 4]} : tensor<3x4xf16> -> tensor<1x3x4xf16>
  // CHECK:     return [[RES]]
}

// -----

// CHECK-LABEL: @DecomposeWACPatternWithPostFP16FC
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf16>)
func.func @DecomposeWACPatternWithPostFP16FC(%act : tensor<1x3x4xf16>) -> tensor<1x3x4xf16> {
  %weights = const.Declare tensor<4x2x2xf16> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xui8>, [#const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %1 = IE.Subtract(%weights, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf16> -> tensor<3x4xf16>
  %6 = IE.FullyConnected(%5, %3) : tensor<3x4xf16>, tensor<4x4xf16> -> tensor<3x4xf16>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf16> -> tensor<1x3x4xf16>
  return %7 : tensor<1x3x4xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK-DAG: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x2xf16> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xui8>, [#const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  // CHECK:     [[MULTIPLY_1:%.+]] = IE.Multiply([[WEIGHTS]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK:     [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY_1]])
  // CHECK:     [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf16> -> tensor<3x4xf16>
  // CHECK:     [[FC_1:%.+]] = IE.FullyConnected([[RESHAPE_2]], [[RESHAPE_1]]) : tensor<3x4xf16>, tensor<4x4xf16> -> tensor<3x4xf16>
  // CHECK:     [[RESHAPE_3:%.+]] = IE.Reshape([[RESHAPE_2]]) {shape_value = [3, 2, 2]} : tensor<3x4xf16> -> tensor<3x2x2xf16>
  // CHECK:     [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE_3]]) {axes_value = [2]} : tensor<3x2x2xf16> -> tensor<3x2xf16>
  // CHECK:     [[MULTIPLY_2:%.+]] = IE.Multiply([[SCALE]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x1xf16>
  // CHECK:     [[RESHAPE_4:%.+]] = IE.Reshape([[MULTIPLY_2]]) {shape_value = [4, 2]} : tensor<4x2x1xf16> -> tensor<4x2xf16>
  // CHECK:     [[FC_2:%.+]]  = IE.FullyConnected([[REDUCE_SUM]], [[RESHAPE_4]]) : tensor<3x2xf16>, tensor<4x2xf16> -> tensor<3x4xf16>
  // CHECK:     [[SUBTRACT:%.+]]  = IE.Subtract([[FC_1]], [[FC_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x4xf16>, tensor<3x4xf16> -> tensor<3x4xf16>
  // CHECK:     [[RES:%.+]]  = IE.Reshape([[SUBTRACT]]) {shape_value = [1, 3, 4]} : tensor<3x4xf16> -> tensor<1x3x4xf16>
  // CHECK:     return [[RES]]
}

// -----

// CHECK-LABEL: @DecomposeWAIStaticScalePatternWithPostFP32MatMul
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf32>, [[WEIGHTS:%.+]]: tensor<4x2x2xui8>)
func.func @DecomposeWAIStaticScalePatternWithPostFP32MatMul(%act : tensor<1x3x4xf32>, %weights: tensor<4x2x2xui8>) -> tensor<1x3x4xf32> {
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.MatMul(%5, %4) {transpose_b} : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %7 : tensor<1x3x4xf32>

  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  // CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[CONVERT_1]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY_1]])
  // CHECK: [[CONVERT_2:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[RESHAPE_2]], [[CONVERT_2]]) {transpose_b} : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[RESHAPE_2]]) {shape_value = [3, 2, 2]} : tensor<3x4xf32> -> tensor<3x2x2xf32>
  // CHECK: [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE_3]]) {axes_value = [2]} : tensor<3x2x2xf32> -> tensor<3x2xf32>
  // CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[SCALE]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x1xf16>
  // CHECK: [[RESHAPE_4:%.+]] = IE.Reshape([[MULTIPLY_2]]) {shape_value = [4, 2]} : tensor<4x2x1xf16> -> tensor<4x2xf16>
  // CHECK: [[CONVERT_3:%.+]]  = IE.Convert([[RESHAPE_4]]) {dstElemType = f32} : tensor<4x2xf16> -> tensor<4x2xf32>
  // CHECK: [[MATMUL_2:%.+]]  = IE.MatMul([[REDUCE_SUM]], [[CONVERT_3]]) {transpose_b} : tensor<3x2xf32>, tensor<4x2xf32> -> tensor<3x4xf32>
  // CHECK: [[SUBTRACT:%.+]]  = IE.Subtract([[MATMUL_1]], [[MATMUL_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x4xf32>, tensor<3x4xf32> -> tensor<3x4xf32>
  // CHECK: [[RES:%.+]]  = IE.Reshape([[SUBTRACT]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @DecomposeWAIStaticScalePatternWithPostFP32FC
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf32>, [[WEIGHTS:%.+]]: tensor<4x2x2xui8>)
func.func @DecomposeWAIStaticScalePatternWithPostFP32FC(%act : tensor<1x3x4xf32>, %weights: tensor<4x2x2xui8>) -> tensor<1x3x4xf32> {
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.FullyConnected(%5, %4) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %7 : tensor<1x3x4xf32>

  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  // CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[CONVERT_1]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY_1]])
  // CHECK: [[CONVERT_2:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK: [[FC_1:%.+]] = IE.FullyConnected([[RESHAPE_2]], [[CONVERT_2]]) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[RESHAPE_2]]) {shape_value = [3, 2, 2]} : tensor<3x4xf32> -> tensor<3x2x2xf32>
  // CHECK: [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE_3]]) {axes_value = [2]} : tensor<3x2x2xf32> -> tensor<3x2xf32>
  // CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[SCALE]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x1xf16>
  // CHECK: [[RESHAPE_4:%.+]] = IE.Reshape([[MULTIPLY_2]]) {shape_value = [4, 2]} : tensor<4x2x1xf16> -> tensor<4x2xf16>
  // CHECK: [[CONVERT_3:%.+]]  = IE.Convert([[RESHAPE_4]]) {dstElemType = f32} : tensor<4x2xf16> -> tensor<4x2xf32>
  // CHECK: [[FC_2:%.+]]  = IE.FullyConnected([[REDUCE_SUM]], [[CONVERT_3]]) : tensor<3x2xf32>, tensor<4x2xf32> -> tensor<3x4xf32>
  // CHECK: [[SUBTRACT:%.+]]  = IE.Subtract([[FC_1]], [[FC_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x4xf32>, tensor<3x4xf32> -> tensor<3x4xf32>
  // CHECK: [[RES:%.+]]  = IE.Reshape([[SUBTRACT]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @DecomposeWAIStaticScalePatternWithPostFP16MatMul
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf16>, [[WEIGHTS:%.+]]: tensor<4x2x2xui8>)
func.func @DecomposeWAIStaticScalePatternWithPostFP16MatMul(%act : tensor<1x3x4xf16>, %weights: tensor<4x2x2xui8>) -> tensor<1x3x4xf16> {
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf16> -> tensor<3x4xf16>
  %6 = IE.MatMul(%5, %3) {transpose_b} : tensor<3x4xf16>, tensor<4x4xf16> -> tensor<3x4xf16>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf16> -> tensor<1x3x4xf16>
  return %7 : tensor<1x3x4xf16>

  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  // CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[CONVERT_1]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY_1]])
  // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf16> -> tensor<3x4xf16>
  // CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[RESHAPE_2]], [[RESHAPE_1]]) {transpose_b} : tensor<3x4xf16>, tensor<4x4xf16> -> tensor<3x4xf16>
  // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[RESHAPE_2]]) {shape_value = [3, 2, 2]} : tensor<3x4xf16> -> tensor<3x2x2xf16>
  // CHECK: [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE_3]]) {axes_value = [2]} : tensor<3x2x2xf16> -> tensor<3x2xf16>
  // CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[SCALE]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x1xf16>
  // CHECK: [[RESHAPE_4:%.+]] = IE.Reshape([[MULTIPLY_2]]) {shape_value = [4, 2]} : tensor<4x2x1xf16> -> tensor<4x2xf16>
  // CHECK: [[MATMUL_2:%.+]]  = IE.MatMul([[REDUCE_SUM]], [[RESHAPE_4]]) {transpose_b} : tensor<3x2xf16>, tensor<4x2xf16> -> tensor<3x4xf16>
  // CHECK: [[SUBTRACT:%.+]]  = IE.Subtract([[MATMUL_1]], [[MATMUL_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x4xf16>, tensor<3x4xf16> -> tensor<3x4xf16>
  // CHECK: [[RES:%.+]]  = IE.Reshape([[SUBTRACT]]) {shape_value = [1, 3, 4]} : tensor<3x4xf16> -> tensor<1x3x4xf16>
  // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @DecomposeWAIStaticScalePatternWithPostFP16FC
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf16>, [[WEIGHTS:%.+]]: tensor<4x2x2xui8>)
func.func @DecomposeWAIStaticScalePatternWithPostFP16FC(%act : tensor<1x3x4xf16>, %weights: tensor<4x2x2xui8>) -> tensor<1x3x4xf16> {
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf16> -> tensor<3x4xf16>
  %6 = IE.FullyConnected(%5, %3) : tensor<3x4xf16>, tensor<4x4xf16> -> tensor<3x4xf16>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf16> -> tensor<1x3x4xf16>
  return %7 : tensor<1x3x4xf16>

  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  // CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[CONVERT_1]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY_1]])
  // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf16> -> tensor<3x4xf16>
  // CHECK: [[FC_1:%.+]] = IE.FullyConnected([[RESHAPE_2]], [[RESHAPE_1]]) : tensor<3x4xf16>, tensor<4x4xf16> -> tensor<3x4xf16>
  // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[RESHAPE_2]]) {shape_value = [3, 2, 2]} : tensor<3x4xf16> -> tensor<3x2x2xf16>
  // CHECK: [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE_3]]) {axes_value = [2]} : tensor<3x2x2xf16> -> tensor<3x2xf16>
  // CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[SCALE]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x1xf16>
  // CHECK: [[RESHAPE_4:%.+]] = IE.Reshape([[MULTIPLY_2]]) {shape_value = [4, 2]} : tensor<4x2x1xf16> -> tensor<4x2xf16>
  // CHECK: [[FC_2:%.+]]  = IE.FullyConnected([[REDUCE_SUM]], [[RESHAPE_4]]) : tensor<3x2xf16>, tensor<4x2xf16> -> tensor<3x4xf16>
  // CHECK: [[SUBTRACT:%.+]]  = IE.Subtract([[FC_1]], [[FC_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x4xf16>, tensor<3x4xf16> -> tensor<3x4xf16>
  // CHECK: [[RES:%.+]]  = IE.Reshape([[SUBTRACT]]) {shape_value = [1, 3, 4]} : tensor<3x4xf16> -> tensor<1x3x4xf16>
  // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @NotDecomposePatternIfWeightsHasMultiUsers
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf32>, [[WEIGHTS:%.+]]: tensor<4x2x2xui8>)
func.func @NotDecomposePatternIfWeightsHasMultiUsers(%act : tensor<1x3x4xf32>, %weights: tensor<4x2x2xui8>) -> (tensor<4x4xf32>, tensor<1x3x4xf32>) {
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<1x2x1xf16> = dense<[0, 1]> : tensor<2xui8>, [#const.Reshape<[1, 2, 1]>, #const.CastElemType<f16>]
  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<1x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.FullyConnected(%5, %4) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %4, %7 : tensor<4x4xf32>, tensor<1x3x4xf32>

  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK: [[SHIFT:%.+]] = const.Declare tensor<1x2x1xf16> = dense<[0, 1]> : tensor<2xui8>, [#const.Reshape<[1, 2, 1]>, #const.CastElemType<f16>]
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  // CHECK: [[SUBTRACT:%.+]] = IE.Subtract([[CONVERT_1]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<1x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[SUBTRACT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY]])
  // CHECK: [[CONVERT_2:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK: [[FC:%.+]] = IE.FullyConnected([[RESHAPE_2]], [[CONVERT_2]]) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK: return [[CONVERT_2]], [[RESHAPE_3]]
}

// -----

// CHECK-LABEL: @NotDecomposePatternIfNotPerChannelZP
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf32>, [[WEIGHTS:%.+]]: tensor<4x2x2xui8>)
func.func @NotDecomposePatternIfNotPerChannelZP(%act : tensor<1x3x4xf32>, %weights: tensor<4x2x2xui8>) -> tensor<1x3x4xf32> {
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<1x2x1xf16> = dense<[0, 1]> : tensor<2xui8>, [#const.Reshape<[1, 2, 1]>, #const.CastElemType<f16>]
  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<1x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.FullyConnected(%5, %4) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %7 : tensor<1x3x4xf32>

  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK: [[SHIFT:%.+]] = const.Declare tensor<1x2x1xf16> = dense<[0, 1]> : tensor<2xui8>, [#const.Reshape<[1, 2, 1]>, #const.CastElemType<f16>]
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  // CHECK: [[SUBTRACT:%.+]] = IE.Subtract([[CONVERT_1]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<1x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[SUBTRACT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY]])
  // CHECK: [[CONVERT_2:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK: [[FC:%.+]] = IE.FullyConnected([[RESHAPE_2]], [[CONVERT_2]]) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK: return [[RESHAPE_3]]
}

// -----

// CHECK-LABEL: @NotDecomposePatternForDynamicCase
// CHECK-SAME:  ([[ACT:%.+]]: tensor<1x3x4xf32>, [[WEIGHTS:%.+]]: tensor<4x2x2xui8>, [[SCALE:%.+]]: tensor<4x2x1xf16>)
func.func @NotDecomposePatternForDynamicCase(%act : tensor<1x3x4xf32>, %weights: tensor<4x2x2xui8>, %scale: tensor<4x2x1xf16>) -> tensor<1x3x4xf32> {
  %shift = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.FullyConnected(%5, %4) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %7 : tensor<1x3x4xf32>

  // CHECK: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[0, 1, 2, 1, 2, 0, 2, 2]> : tensor<8xui8>, [#const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16} : tensor<4x2x2xui8> -> tensor<4x2x2xf16>
  // CHECK: [[SUBTRACT:%.+]] = IE.Subtract([[CONVERT_1]], [[SHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[SUBTRACT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[MULTIPLY]])
  // CHECK: [[CONVERT_2:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK: [[FC:%.+]] = IE.FullyConnected([[RESHAPE_2]], [[CONVERT_2]]) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK: return [[RESHAPE_3]]
}
