//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u16:f16, 0.0025215686274509803>

// CHECK-LABEL: @AvoidMixedPrecisionConv16BitQuantize
func.func @AvoidMixedPrecisionConv16BitQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>
IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME: : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>
    // CHECK:   IE.Quantize([[CONV]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
}
