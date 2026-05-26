//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --legalize-convbackpropdata --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010
// COM: F8 is only supported on NPU50+, no need to run these tests on all platforms.

// CHECK-LABEL: @ConvertQuantizedGroupConvBackpropDataToGroupTransposedConvF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x23x30xf16>
func.func @ConvertQuantizedGroupConvBackpropDataToGroupTransposedConvF8E4M3FN(%input: tensor<1x32x23x30xf16>) -> tensor<1x64x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    %fq = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<2x16x32x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<2x16x32x2x1xf16>

    %output = IE.GroupConvolutionBackpropData(%fq, %filter_fq) {
        dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>
    // CHECK:    [[FILTER_FQ:%.+]] = IE.FakeQuantize([[FILTER]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<2x32x16x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<2x32x16x2x1xf16>

    // CHECK:    [[CONV:%.+]] = IE.GroupTransposedConvolution([[FQ]], [[FILTER_FQ]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x64x46x59xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @ConvertDequantizedGroupConvBackpropDataToGroupTransposedConvF8E4M3FNPerTensor
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x23x30x!qElemType>
func.func @ConvertDequantizedGroupConvBackpropDataToGroupTransposedConvF8E4M3FNPerTensor(%input: tensor<1x32x23x30x!qElemType>) -> tensor<1x64x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    %dq = IE.Dequantize(%input) {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<1x32x23x30x!qElemType> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1x!qElemType> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %filter_dq = IE.Dequantize(%filter) {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<2x16x32x2x1x!qElemType> -> tensor<2x16x32x2x1xf16>

    %output = IE.GroupConvolutionBackpropData(%dq, %filter_dq) {
        dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1x!qElemType> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[DQ:%.+]] = IE.Dequantize([[INPUT]])
    // CHECK-SAME:  {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<1x32x23x30x!qElemType> -> tensor<1x32x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.Dequantize([[FILTER]])
    // CHECK-SAME:  {dstElemType = f16} : tensor<2x32x16x2x1x!qElemType> -> tensor<2x32x16x2x1xf16>

    // CHECK:    [[CONV:%.+]] = IE.GroupTransposedConvolution([[DQ]], [[FILTER_DQ]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x64x46x59xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>
!qElemType1 = !quant.uniform<u8:f16:2, {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0}>

// CHECK-LABEL: @ConvertDequantizedGroupConvBackpropDataToGroupTransposedConvF8E4M3FNPerAxis
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x23x30x!qElemType>
func.func @ConvertDequantizedGroupConvBackpropDataToGroupTransposedConvF8E4M3FNPerAxis(%input: tensor<1x32x23x30x!qElemType>) -> tensor<1x64x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    %dq = IE.Dequantize(%input) {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<1x32x23x30x!qElemType> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1x!qElemType1> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
    %filter_dq = IE.Dequantize(%filter) {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<2x16x32x2x1x!qElemType1> -> tensor<2x16x32x2x1xf16>

    %output = IE.GroupConvolutionBackpropData(%dq, %filter_dq) {
        dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1x!qElemType1> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.CastElemType<ui8>, #const.Reverse<2 : i64>, #const.Transpose<#map>, #const.CastElemType<!qElemType1>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[DQ:%.+]] = IE.Dequantize([[INPUT]])
    // CHECK-SAME:  {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<1x32x23x30x!qElemType> -> tensor<1x32x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.Dequantize([[FILTER]])
    // CHECK-SAME:  {dstElemType = f16} : tensor<2x32x16x2x1x!qElemType1> -> tensor<2x32x16x2x1xf16>

    // CHECK:    [[CONV:%.+]] = IE.GroupTransposedConvolution([[DQ]], [[FILTER_DQ]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x64x46x59xf16>
}

// -----

// CHECK-LABEL: @ConvertQuantizedConvertedConvBackpropDataToTransposedConvF8E5M2
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30xf16>)
func.func @ConvertQuantizedConvertedConvBackpropDataToTransposedConvF8E5M2(%input: tensor<1x16x23x30xf16>) -> tensor<1x32x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    %fq = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<16x32x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_fq) {dstElemType = f32} : tensor<16x32x2x1xf16> -> tensor<16x32x2x1xf32>

    %output = IE.ConvolutionBackpropData(%fq, %filter_convert) {
        dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf32> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>
    // CHECK:    [[FILTER_FQ:%.+]] = IE.FakeQuantize([[FILTER]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<32x16x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<32x16x2x1xf16>
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[FILTER_FQ]]) {dstElemType = f32} : tensor<32x16x2x1xf16> -> tensor<32x16x2x1xf32>

    // CHECK:    [[CONV:%.+]] = IE.TransposedConvolution([[FQ]], [[CONVERT]])
    // CHECK-SAME:  {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf32> -> tensor<1x32x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x32x46x59xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @ConvertDequantizedConvertedConvBackpropDataToTransposedConvF8E5M2PerTensor
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30x!qElemType>)
func.func @ConvertDequantizedConvertedConvBackpropDataToTransposedConvF8E5M2PerTensor(%input: tensor<1x16x23x30x!qElemType>) -> tensor<1x32x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    %dq = IE.Dequantize(%input) {dstElemType = f16, low_fp_type = f8E5M2} : tensor<1x16x23x30x!qElemType> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1x!qElemType> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %filter_dq = IE.Dequantize(%filter) {dstElemType = f16, low_fp_type = f8E5M2} : tensor<16x32x2x1x!qElemType> -> tensor<16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_dq) {dstElemType = f32} : tensor<16x32x2x1xf16> -> tensor<16x32x2x1xf32>

    %output = IE.ConvolutionBackpropData(%dq, %filter_convert) {
        dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf32> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<32x16x2x1x!qElemType> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[DQ:%.+]] = IE.Dequantize([[INPUT]])
    // CHECK-SAME:  {dstElemType = f16, low_fp_type = f8E5M2} : tensor<1x16x23x30x!qElemType> -> tensor<1x16x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.Dequantize([[FILTER]])
    // CHECK-SAME:  {dstElemType = f16} : tensor<32x16x2x1x!qElemType> -> tensor<32x16x2x1xf16>
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[FILTER_DQ]]) {dstElemType = f32} : tensor<32x16x2x1xf16> -> tensor<32x16x2x1xf32>

    // CHECK:    [[CONV:%.+]] = IE.TransposedConvolution([[DQ]], [[CONVERT]])
    // CHECK-SAME:  {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf32> -> tensor<1x32x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x32x46x59xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>
!qElemType1 = !quant.uniform<u8:f16:1, {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0}>

// CHECK-LABEL: @ConvertDequantizedConvertedConvBackpropDataToTransposedConvF8E5M2PerAxis
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30x!qElemType>)
func.func @ConvertDequantizedConvertedConvBackpropDataToTransposedConvF8E5M2PerAxis(%input: tensor<1x16x23x30x!qElemType>) -> tensor<1x32x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    %dq = IE.Dequantize(%input) {dstElemType = f16, low_fp_type = f8E5M2} : tensor<1x16x23x30x!qElemType> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1x!qElemType1> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
    %filter_dq = IE.Dequantize(%filter) {dstElemType = f16, low_fp_type = f8E5M2} : tensor<16x32x2x1x!qElemType1> -> tensor<16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_dq) {dstElemType = f32} : tensor<16x32x2x1xf16> -> tensor<16x32x2x1xf32>

    %output = IE.ConvolutionBackpropData(%dq, %filter_convert) {
        dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf32> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<32x16x2x1x!qElemType1> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.CastElemType<ui8>, #const.Reverse<1 : i64>, #const.Transpose<#map>, #const.CastElemType<!qElemType1>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[DQ:%.+]] = IE.Dequantize([[INPUT]])
    // CHECK-SAME:  {dstElemType = f16, low_fp_type = f8E5M2} : tensor<1x16x23x30x!qElemType> -> tensor<1x16x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.Dequantize([[FILTER]])
    // CHECK-SAME:  {dstElemType = f16} : tensor<32x16x2x1x!qElemType1> -> tensor<32x16x2x1xf16>
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[FILTER_DQ]]) {dstElemType = f32} : tensor<32x16x2x1xf16> -> tensor<32x16x2x1xf32>

    // CHECK:    [[CONV:%.+]] = IE.TransposedConvolution([[DQ]], [[CONVERT]])
    // CHECK-SAME:  {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf32> -> tensor<1x32x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x32x46x59xf16>
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeConvBackpropDataTo3x3SplitConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x64x65xf32>)
func.func @LegalizeConvBackpropDataTo3x3SplitConv(%arg0: tensor<1x1x64x65xf32>) -> tensor<1x1x128x130xf32> {
    %cst = const.Declare tensor<1x1x4x4xf32> = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]]]]> : tensor<1x1x4x4xf32>
    %0 = IE.ConvolutionBackpropData(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x1x64x65xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x128x130xf32>
    return %0 : tensor<1x1x128x130xf32>


    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x3x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[1.600000e+01, 1.400000e+01, 0.000000e+00], [8.000000e+00, 6.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x3x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.500000e+01, 1.300000e+01], [0.000000e+00, 7.000000e+00, 5.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x3x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 0.000000e+00, 0.000000e+00], [1.200000e+01, 1.000000e+01, 0.000000e+00], [4.000000e+00, 2.000000e+00, 0.000000e+00]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x3x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.100000e+01, 9.000000e+00], [0.000000e+00, 3.000000e+00, 1.000000e+00]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_3]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_4]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32> -> tensor<1x4x64x65xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x64x65xf32> -> tensor<1x1x128x130xf32>

    // CHECK: return [[D2S]] : tensor<1x1x128x130xf32>
  }
