//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-weights-to-i8 --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>

// We don't convert u8 to i8 because of the zero point value of U8 which must be 128.
// Conversion converts from u8 ZP = 128 to i8 ZP = 0
// CHECK-LABEL: @NotConvertU8Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @NotConvertU8Weights(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %wgt = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<9.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %dequant = IE.Dequantize(%wgt) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %conv = IE.Convolution(%arg0, %dequant) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %conv : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<9.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>

// We don't convert u8 to i8 because of the zero point value of U8 which must be 128.
// Conversion converts from u8 ZP = 128 to i8 ZP = 0
// CHECK-LABEL: @NotConvertU8WeightsFusedDequant
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @NotConvertU8WeightsFusedDequant(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %wgt = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<9.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %conv = IE.Convolution(%arg0, %wgt) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14xf16>
    return %conv : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<9.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[CONV]]
}
