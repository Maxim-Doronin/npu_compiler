//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_0To175_Sigmoid
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_0To175_Sigmoid(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 1.750000e+02 : f64},
        post_op = #IE.Sigmoid<>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <SIGMOID>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.750000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_32To192
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_32To192(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 3.200000e+01 : f64, max = 1.920000e+02 : f64},
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LRELUX>,
    // CHECK-SAME:          clamp_low = 3.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.920000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_0To175_Swish
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_0To175_Swish(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 1.750000e+02 : f64},
        post_op = #IE.Swish<beta=1.5>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <SWISH>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.750000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_0To64_Gelu
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_0To64_Gelu(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 6.400000e+01 : f64},
        post_op = #IE.Gelu<>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <GELU>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 6.400000e+01 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_0To64_HSwish
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_0To64_HSwish(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 6.400000e+01 : f64},
        post_op = #IE.HSwish<>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <HSWISH>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 6.400000e+01 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}
