//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @Conv2dLeakyReluWithQuantize
func.func @Conv2dLeakyReluWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x3x3x3x!qElemType> {
    %cst = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> : tensor<3x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x3x3x3xf16> -> tensor<1x3x3x3x!qElemType>

    return %1 : tensor<1x3x3x3x!qElemType>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<3x16x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x3x3x3x!qElemType>
}

// -----

!qElemType = !quant.quantile<u4:u8:f16, {0.0,16.0,32.0,48.0,64.0,80.0,96.0,112.0,128.0,144.0,160.0,176.0,192.0,208.0,224.0,240.0}:1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK: !qElemType = !quant.quantile<u4:u8:f16, {0.000000e+00,1.600000e+01,3.200000e+01,4.800000e+01,6.400000e+01,8.000000e+01,9.600000e+01,1.120000e+02,1.280000e+02,1.440000e+02,1.600000e+02,1.760000e+02,1.920000e+02,2.080000e+02,2.240000e+02,2.400000e+02}:1.1534313725490195:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @MixedPrecisionConvQuantile
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x1x1xf16>)
func.func @MixedPrecisionConvQuantile(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>

  return %4 : tensor<1x16x1x1xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType1>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  //CHECK: return [[CONV]]
}
