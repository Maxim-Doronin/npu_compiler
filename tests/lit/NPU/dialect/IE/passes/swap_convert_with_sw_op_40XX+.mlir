//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-convert-with-sw-op %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SwapConvert
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @SwapConvert(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x1x64x256000xf32> {
  %cst = const.Declare tensor<2000x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2000x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2000x8x1x2xf16, {order = #NHWC}> -> tensor<1x2000x128x64xf16, {order = #NHWC}>
  %1 = IE.Reshape(%0) { shape_value = [1, 256000, 1, 64] } : tensor<1x2000x128x64xf16, {order = #NHWC}> -> tensor<1x256000x1x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256000x1x64xf16> -> tensor<1x1x64x256000xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x64x256000xf16> -> tensor<1x1x64x256000xf32>
  return %3 : tensor<1x1x64x256000xf32>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2000x8x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]])
  // CHECK-SAME:        -> tensor<1x2000x128x64xf16, {order = #NHWC}>
  
  // CHECK:       [[RET:%.+]] = IE.Convert([[CONV_RET]]) {dstElemType = f32}
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[RET]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       return [[TRANSPOSE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SmallSizeConv
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @SmallSizeConv(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x1x64x256xf32> {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Reshape(%0) { shape_value = [1, 256, 1, 64] } : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x256x1x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x1x64xf16> -> tensor<1x1x64x256xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x64x256xf16> -> tensor<1x1x64x256xf32>
  return %3 : tensor<1x1x64x256xf32>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]])
  // CHECK-SAME:        -> tensor<1x2x128x64xf16, {order = #NHWC}>
  
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[CONV_RET]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x1x64x256xf32
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvNotSingleUser
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @ConvNotSingleUser(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>)
    -> (tensor<1x1x64x256xf32>, tensor<1x2x128x64xf16, {order = #NHWC}>) {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.25000e+00> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}>
    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Reshape(%0) { shape_value = [1, 256, 1, 64] } : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x256x1x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x1x64xf16> -> tensor<1x1x64x256xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x64x256xf16> -> tensor<1x1x64x256xf32>
  return %3, %0 : tensor<1x1x64x256xf32>, tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], %{{.+}})
  // CHECK-SAME:    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[CONV]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x1x64x256xf32
  // CHECK:       return [[RET]], [[CONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NonDPUParentOp
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x128x64xf16, {order = #NHWC}>)
func.func @NonDPUParentOp(%arg0: tensor<1x2x128x64xf16, {order = #NHWC}>)
    -> tensor<1x1x64x256xf32> {
  %0 = IE.SoftMax(%arg0) {axisInd = 2} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Reshape(%0) { shape_value = [1, 256, 1, 64] } : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x256x1x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x1x64xf16> -> tensor<1x1x64x256xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x64x256xf16> -> tensor<1x1x64x256xf32>
  return %3 : tensor<1x1x64x256xf32>

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]])
  // CHECK-SAME:    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SOFTMAX]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x1x64x256xf32>
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NoSwapInterpolate
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @NoSwapInterpolate(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2048x1x256xf32> {
  %0 = const.Declare tensor<4xsi64> = dense<[1, 8, 256, 256]> : tensor<4xsi64>
  %1 = const.Declare tensor<4xf32>  = dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
  %2 = IE.Interpolate(%arg0, %0, %1) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : 
    tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<4xsi64>, tensor<4xf32> -> tensor<1x8x256x256xf16, {order = #NHWC}>
  %3 = IE.Reshape(%2) { shape_value = [1, 2048, 1, 256] } : tensor<1x8x256x256xf16, {order = #NHWC}> -> tensor<1x2048x1x256xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x2048x1x256xf16> -> tensor<1x2048x1x256xf32>
  return %4 : tensor<1x2048x1x256xf32>

  // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT]]
  // CHECK-SAME:    -> tensor<1x8x256x256xf16, {order = #NHWC}>
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INTERPOLATE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[RESHAPE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x2048x1x256xf32>
  // CHECK:       return [[RET]]
}
