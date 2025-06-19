//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-op-with-quantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType = !quant.uniform<u8:f16, 0.956:128>

// CHECK-LABEL: @PerTensor
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xui8>
func.func @PerTensor(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    //CHECK: [[VAL0:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType} :
    //CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    //CHECK: return [[VAL1]] : tensor<1x3x16x16xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16:1, {1.000000e+00,1.000000e+00,1.000000e+00}>
!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK-LABEL: @PerAxis
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xui8>
func.func @PerAxis(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    //CHECK: [[VAL0:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType} :
    //CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    //CHECK: return [[VAL1]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0:128>

// CHECK-LABEL: @PerTensorDequantizeConvert
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16x!qElemType>
func.func @PerTensorDequantizeConvert(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xui8> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %1 : tensor<1x3x16x16xui8>

    //CHECK: [[VAL0:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = ui8} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xui8>

    //CHECK: return [[VAL0]] : tensor<1x3x16x16xui8>
}


// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK-LABEL: @PerAxisDequantizeConvert
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16x!qElemType>
func.func @PerAxisDequantizeConvert(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xui8> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %1 : tensor<1x3x16x16xui8>

    //CHECK: [[VAL0:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = ui8} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xui8>

    //CHECK: return [[VAL0]] : tensor<1x3x16x16xui8>
}


// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @PerTensorSI8
// CHECK-SAME: [[ARG:%.+]]: tensor<1x12x19x19x!qElemType>
func.func @PerTensorSI8(%arg0: tensor<1x12x19x19x!qElemType>) -> tensor<1x12x19x19xsi8> {
   %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x12x19x19x!qElemType> -> tensor<1x12x19x19xf16>
   %1 = IE.Convert(%0) {dstElemType = si8} : tensor<1x12x19x19xf16> -> tensor<1x12x19x19xsi8>

   return %1 : tensor<1x12x19x19xsi8>

   //CHECK: [[VAL0:%.+]] IE.Dequantize([[ARG]]) {dstElemType = f16} :
   //CHECK-SAME: tensor<1x12x19x19x!qElemType> -> tensor<1x12x19x19xf16>
   //CHECK: [[VAL1:%.+]]  = IE.Convert(%0) {dstElemType = si8} :
   //CHECK-SAME: tensor<1x12x19x19xf16> -> tensor<1x12x19x19xsi8>
   //CHECK: return [[VAL1]] :  tensor<1x12x19x19xsi8>

}


// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>
!qElemType1 = !quant.uniform<u8:f16, 2.000000e+00:128>
// CHECK-LABEL: @FuseWithMultiply
// CHECK-SAME: [[ARG:%.+]]: tensor<1x12x19x19x!qElemType>
func.func @FuseWithMultiply(%arg0: tensor<1x12x19x19x!qElemType>) -> tensor<1x12x19x19xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x12x19x19x!qElemType> -> tensor<1x12x19x19xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x19x19xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x19x19xf16>

    return %1 : tensor<1x12x19x19xf16>

    //CHECK: [[QCAST:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType1} : tensor<1x12x19x19x!qElemType> -> tensor<1x12x19x19x!qElemType1>
    //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QCAST]]) {dstElemType = f16} : tensor<1x12x19x19x!qElemType1> -> tensor<1x12x19x19xf16>

   //CHECK: return [[DEQUANT]] :  tensor<1x12x19x19xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {2.000000e-01:128,4.000000e-01:128,6.000000e-01:128}>
// CHECK-LABEL: @FuseWithDWConvPerAxisQuant
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x19x19x!qElemType>
func.func @FuseWithDWConvPerAxisQuant(%arg0: tensor<1x3x19x19x!qElemType>) -> tensor<1x3x19x19xf16> {
    %cst = const.Declare tensor<3x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf32> isSplat, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 3 : i64>, #const.Reshape<[3, 1, 1, 1]>]
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x19x19x!qElemType> -> tensor<1x3x19x19xf16>
    %1 = IE.GroupConvolution(%0, %cst) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x19x19xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x19x19xf16>

    return %1 : tensor<1x3x19x19xf16>

    //CHECK: [[QCAST:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType1} : tensor<1x3x19x19x!qElemType> -> tensor<1x3x19x19x!qElemType1>
    //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QCAST]]) {dstElemType = f16} : tensor<1x3x19x19x!qElemType1> -> tensor<1x3x19x19xf16>

    //CHECK: return [[DEQUANT]] :  tensor<1x3x19x19xf16>

}

// -----

!qElemType = !quant.quantile<u4:f8E4M3FN:f16, {-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00}:3.000000e+00>
!qElemType1 = !quant.quantile<u4:f8E4M3FN:f16, {-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00}:6.000000e+00>
// CHECK-LABEL: @FuseWithMultiplyPerTensorQuantile
// CHECK-SAME: [[ARG:%.+]]: tensor<1x2x19x19x!qElemType>
func.func @FuseWithMultiplyPerTensorQuantile(%arg0: tensor<1x2x19x19x!qElemType>) -> tensor<1x2x19x19xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x19x19x!qElemType> -> tensor<1x2x19x19xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x2x19x19xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x19x19xf16>

    return %1 : tensor<1x2x19x19xf16>

    //CHECK: [[QCAST:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType1} : tensor<1x2x19x19x!qElemType> -> tensor<1x2x19x19x!qElemType1>
    //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QCAST]]) {dstElemType = f16} : tensor<1x2x19x19x!qElemType1> -> tensor<1x2x19x19xf16>

   //CHECK: return [[DEQUANT]] :  tensor<1x2x19x19xf16>
}


// -----

!qElemType = !quant.quantile<u4:f8E4M3FN:f16:1, {-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00}:{2.000000e+00,3.000000e+00}>
!qElemType1 = !quant.quantile<u4:f8E4M3FN:f16:1, {-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00}:{4.000000e+00,6.000000e+00}>
// CHECK-LABEL: @FuseWithMultiplyPerAxisQuantile
// CHECK-SAME: [[ARG:%.+]]: tensor<1x2x19x19x!qElemType>
func.func @FuseWithMultiplyPerAxisQuantile(%arg0: tensor<1x2x19x19x!qElemType>) -> tensor<1x2x19x19xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x19x19x!qElemType> -> tensor<1x2x19x19xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x2x19x19xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x19x19xf16>

    return %1 : tensor<1x2x19x19xf16>

    //CHECK: [[QCAST:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType1} : tensor<1x2x19x19x!qElemType> -> tensor<1x2x19x19x!qElemType1>
    //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QCAST]]) {dstElemType = f16} : tensor<1x2x19x19x!qElemType1> -> tensor<1x2x19x19xf16>

   //CHECK: return [[DEQUANT]] :  tensor<1x2x19x19xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128}>

// CHECK-LABEL: @NotFuseWithDWConvPrecisionChange
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x19x19x!qElemType>
func.func @NotFuseWithDWConvPrecisionChange(%arg0: tensor<1x3x19x19x!qElemType>) -> tensor<1x3x19x19xf32> {
    %cst = const.Declare tensor<3x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf32> isSplat, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 3 : i64>, #const.Reshape<[3, 1, 1, 1]>]
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x19x19x!qElemType> -> tensor<1x3x19x19xf16>
    %1 = IE.GroupConvolution(%0, %cst) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x19x19xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x19x19xf32>

    return %1 : tensor<1x3x19x19xf32>

    //CHECK: [[DWCONV:%.+]] = IE.GroupConvolution

    //CHECK: return [[DWCONV]] :  tensor<1x3x19x19xf32>

}


// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128}>

// CHECK-LABEL: @NotFuseWithDWConvHasBias
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x19x19x!qElemType>
func.func @NotFuseWithDWConvHasBias(%arg0: tensor<1x3x19x19x!qElemType>) -> tensor<1x3x19x19xf16> {
    %cst = const.Declare tensor<3x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf32> isSplat, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 3 : i64>, #const.Reshape<[3, 1, 1, 1]>]
    %cst1 = const.Declare tensor<3x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf32> isSplat, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 3 : i64>, #const.Reshape<[3, 1, 1, 1]>]
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x19x19x!qElemType> -> tensor<1x3x19x19xf16>
    %1 = IE.GroupConvolution(%0, %cst, %cst1) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x19x19xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x19x19xf16>

    return %1 : tensor<1x3x19x19xf16>

    //CHECK: [[DWCONV:%.+]] = IE.GroupConvolution

    //CHECK: return [[DWCONV]] :  tensor<1x3x19x19xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128}>

// CHECK-LABEL: @NotFuseWithDWConvHasPadding
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x19x19x!qElemType>
func.func @NotFuseWithDWConvHasPadding(%arg0: tensor<1x3x19x19x!qElemType>) -> tensor<1x3x20x19xf16> {
    %cst = const.Declare tensor<3x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf32> isSplat, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 3 : i64>, #const.Reshape<[3, 1, 1, 1]>]
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x19x19x!qElemType> -> tensor<1x3x19x19xf16>
    %1 = IE.GroupConvolution(%0, %cst) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x19x19xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x20x19xf16>

    return %1 : tensor<1x3x20x19xf16>

    //CHECK: [[DWCONV:%.+]] = IE.GroupConvolution

    //CHECK: return [[DWCONV]] :  tensor<1x3x20x19xf16>

}
