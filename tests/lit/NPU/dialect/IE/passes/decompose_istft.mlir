//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-istft %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @DecomposeISTFT
func.func @DecomposeISTFT(%arg0: tensor<2x3x2xf16>) -> tensor<4xf16> {
    %cst = const.Declare tensor<2xf16> =
        dense<1.0> : tensor<2xf16>
    %cst_0 = const.Declare tensor<1xsi32> =
        dense<2> : tensor<si64>,
          [
            #const.Reshape<[1]>
          ]
    %cst_1 = const.Declare tensor<1xsi32> =
        dense<4> : tensor<si64>,
          [
            #const.Reshape<[1]>
          ]
    %cst_2 = const.Declare tensor<1xsi32> =
        dense<4> : tensor<si64>,
          [
            #const.Reshape<[1]>
          ]
    %0 = IE.ISTFT(%arg0, %cst, %cst_0, %cst_1, %cst_2) : tensor<2x3x2xf16>, tensor<2xf16>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32> -> tensor<4xf16>
    return %0 : tensor<4xf16>


    // CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x2xf16>) -> tensor<4xf16>
    // CHECK:  [[CST:%.+]] = const.Declare tensor<2xf16> = dense<1.000000e+00> : tensor<2xf16>
    // CHECK:  [[CST_0:%.+]] = const.Declare tensor<1xsi32> = dense<2> : tensor<si64>, [#const.Reshape<[1]>]
    // CHECK:  [[CST_1:%.+]] = const.Declare tensor<1xsi32> = dense<4> : tensor<si64>, [#const.Reshape<[1]>]
    // CHECK:  [[CST_2:%.+]] = const.Declare tensor<1xsi32> = dense<4> : tensor<si64>, [#const.Reshape<[1]>]
    // CHECK:  [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 0] [2, 1, 2] : tensor<2x3x2xf16> to tensor<2x1x2xf16>
    // CHECK:  [[SLICE_1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0] [2, 1, 2] : tensor<2x3x2xf16> to tensor<2x1x2xf16>
    // CHECK:  [[SLICE_2:%.+]] = IE.Slice [[ARG0]] [0, 2, 0] [2, 1, 2] : tensor<2x3x2xf16> to tensor<2x1x2xf16>
    // CHECK:  [[RESHAPE:%.+]] = IE.Reshape([[SLICE]]) {shape_value = [2, 2]} : tensor<2x1x2xf16> -> tensor<2x2xf16>
    // CHECK:  [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [2, 2]} : tensor<2x1x2xf16> -> tensor<2x2xf16>
    // CHECK:  [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [2, 2]} : tensor<2x1x2xf16> -> tensor<2x2xf16>
    // CHECK:  [[IRDFT:%.+]] = IE.IRDFT([[RESHAPE]]) {axes_attr = [0], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [2]} : tensor<2x2xf16> -> tensor<2xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[IRDFT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
    // CHECK:  [[IRDFT_1:%.+]] = IE.IRDFT([[RESHAPE_1]]) {axes_attr = [0], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [2]} : tensor<2x2xf16> -> tensor<2xf16>
    // CHECK:  [[MULTIPLY_1:%.+]] = IE.Multiply([[IRDFT_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
    // CHECK:  [[IRDFT_2:%.+]] = IE.IRDFT([[RESHAPE_2]]) {axes_attr = [0], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [2]} : tensor<2x2xf16> -> tensor<2xf16>
    // CHECK:  [[MULTIPLY_2:%.+]] = IE.Multiply([[IRDFT_2]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
    // CHECK:  [[CST_3:%.+]] = const.Declare tensor<4xf16> = dense<0.000000e+00> : tensor<4xf16>
    // CHECK:  [[CST_4:%.+]] = const.Declare tensor<4xf16> = dense<0.000000e+00> : tensor<4xf16>
    // CHECK:  [[PAD:%.+]] = IE.Pad([[MULTIPLY]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0], pads_end_attr = [2]} : tensor<2xf16> -> tensor<4xf16>
    // CHECK:  [[ADD:%.+]] = IE.Add([[CST_3]], [[PAD]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    // CHECK:  [[MULTIPLY_3:%.+]] = IE.Multiply([[CST]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
    // CHECK:  [[PAD_1:%.+]] = IE.Pad([[MULTIPLY_3]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0], pads_end_attr = [2]} : tensor<2xf16> -> tensor<4xf16>
    // CHECK:  [[ADD_1:%.+]] = IE.Add([[CST_4]], [[PAD_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :  tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    // CHECK:  [[CST_5:%.+]] = const.Declare tensor<f16> = dense<1.000170e-04> : tensor<f16>
    // CHECK:  [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :  tensor<4xf16>, tensor<f16> -> tensor<4xf16>
    // CHECK:  [[DIVIDE:%.+]] = IE.Divide([[ADD]], [[ADD_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :  tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    // CHECK:  return [[DIVIDE]] : tensor<4xf16>

}
