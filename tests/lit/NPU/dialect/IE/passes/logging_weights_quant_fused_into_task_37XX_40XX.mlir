//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env OV_NPU_LOG_LEVEL=LOG_TRACE vpux-opt --split-input-file --init-compiler="platform=%platform%" --logging-weights-quant-fused-into-task %s 2>&1 | FileCheck %s
// REQUIRES: dev-build && (platform-NPU3720 || platform-NPU4000)

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
func.func @ConvWithConstQuantWeights(%input: tensor<1x16x16x16xf16>) -> tensor<1x32x16x16xf16>{
    %weights = const.Declare tensor<32x16x1x1x!qElemType> = dense<1.0> : tensor<32x16x1x1xf16>
    %conv = IE.Convolution(%input, %weights){ dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<32x16x1x1x!qElemType> -> tensor<1x32x16x16xf16>
    return %conv : tensor<1x32x16x16xf16>
    //CHECK:  Weights constant(WAC) has quantized element type for NCE op
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
func.func @ConvWithI8InputQuantWeights(%input: tensor<1x16x16x16xf16>, %weights: tensor<32x16x1x1xi8>) -> tensor<1x32x16x16xf16>{
    %weights_reshaped = IE.QuantizeCast(%weights) {dstElemType = !qElemType} : tensor<32x16x1x1xi8> -> tensor<32x16x1x1x!qElemType>
    %conv = IE.Convolution(%input, %weights_reshaped){ dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<32x16x1x1x!qElemType> -> tensor<1x32x16x16xf16>
    return %conv : tensor<1x32x16x16xf16>
    //CHECK:  Weights block argument(WAI) has quantized element type for NCE op
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>
func.func @ConvWithI4InputQuantWeights(%input: tensor<1x16x16x16xf16>, %weights: tensor<32x16x1x1xi4>) -> tensor<1x32x16x16xf16>{
    %weights_reshaped = IE.QuantizeCast(%weights) {dstElemType = !qElemType} : tensor<32x16x1x1xi4> -> tensor<32x16x1x1x!qElemType>
    %conv = IE.Convolution(%input, %weights_reshaped){ dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<32x16x1x1x!qElemType> -> tensor<1x32x16x16xf16>
    return %conv : tensor<1x32x16x16xf16>
    //CHECK:  Weights block argument(WAI) has quantized element type for NCE op
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
func.func @ConvWithViewOpConstQuantWeights(%input: tensor<1x16x16x16xf16>) -> tensor<1x32x16x8xf16>{
    %weights = const.Declare tensor<32x16x3x3x!qElemType> = dense<1.0> : tensor<32x16x3x3xf16>
    %weights_reshaped = IE.Reshape(%weights){shape_value = [32,16,1,9]} : tensor<32x16x3x3x!qElemType> -> tensor<32x16x1x9x!qElemType>
    %conv = IE.Convolution(%input, %weights_reshaped){ dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<32x16x1x9x!qElemType> -> tensor<1x32x16x8xf16>
    return %conv : tensor<1x32x16x8xf16>
    //CHECK:  Weights constant(WAC) has quantized element type for NCE op
}

// -----

func.func @ConvWithViewOpConstFloatWeights(%input: tensor<1x16x16x16xf16>) -> tensor<1x32x16x8xf16>{
    %weights = const.Declare tensor<32x16x3x3xf16> = dense<1.0> : tensor<32x16x3x3xf16>
    %weights_reshaped = IE.Reshape(%weights){shape_value = [32,16,1,9]} : tensor<32x16x3x3xf16> -> tensor<32x16x1x9xf16>
    %conv = IE.Convolution(%input, %weights_reshaped){ dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<32x16x1x9xf16> -> tensor<1x32x16x8xf16>
    return %conv : tensor<1x32x16x8xf16>
    //CHECK-NOT:  Weights constant(WAC) has quantized element type for NCE op
}
