//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: platform-NPU5010


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @dynamicLSTMSequence
// CHECK-SAME: ([[ARG_0:.+]]: tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, [[ARG_1:.+]]: tensor<1x1x1x128xf16>, [[ARG_2:.+]]: tensor<1x1x1x128xf16>) -> (tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
func.func @dynamicLSTMSequence(%arg0: tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x1x1x128xf16>, %arg2: tensor<1x1x1x128xf16>) -> (tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
    %cst = const.Declare tensor<1x4x128x128xf16> = dense<0.000000e+00> : tensor<1x512x128xf32>, [#const.Reshape<[1, 4, 128, 128]>, #const.CastElemType<f16>]
    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %cst) {direction = #IE.rnn_seq_direction<FORWARD>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 0>, useDpu = true} : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16> -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x4x128x128xf16> = dense<0.000000e+00> : tensor<1x512x128xf32>, [#const.Reshape<[1, 4, 128, 128]>, #const.CastElemType<f16>]
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x2x2432xsi32> = dense<0> : tensor<1x1x2x2432xsi32>
    // CHECK: [[OUT_HV:%.+]], [[OUT_HS:%.+]], [[OUT_CS:%.+]] = VPU.LSTMSequence([[ARG_0]], [[ARG_1]], [[ARG_2]], [[CST]], [[CST_0]]) {direction = #IE.rnn_seq_direction<FORWARD>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 1>, useDpu = true} : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16>, tensor<1x1x2x2432xsi32> -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    // CHECK: return [[OUT_HV]], [[OUT_HS]], [[OUT_CS]] : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @FlashSDPA
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x8x32x128xf16, {order = #NCWH}>
func.func @FlashSDPA(%arg0: tensor<1x8x64x64xf16>, %arg1: tensor<1x8x32x64xf16>, %arg2: tensor<1x8x32x128xf16, {order = #NCWH}>) -> tensor<1x8x64x128xf16> {
    %cst_1 = const.Declare tensor<1x8x64x128xf16> = dense<0.000000e+00> : tensor<1x8x64x128xf16>
    %cst_2 = const.Declare tensor<1x1x8x64xf16> = dense<0xFC00> : tensor<1x1x8x64xf16>
    %cst_3 = const.Declare tensor<1x1x8x64xf32> = dense<0.000000e+00> : tensor<1x1x8x64xf32>

    %result_running_output, %result_running_max, %result_running_sum =
        IE.FlashSDPA(%arg0, %arg1, %arg2, %cst_1, %cst_2, %cst_3) {is_head = true, is_tail = true, source_seq_len_pad_size = 0 : i64}
            : tensor<1x8x64x64xf16>, tensor<1x8x32x64xf16>, tensor<1x8x32x128xf16, {order = #NCWH}>, tensor<1x8x64x128xf16>, tensor<1x1x8x64xf16>, tensor<1x1x8x64xf32>
            -> tensor<1x8x64x128xf16>, tensor<1x1x8x64xf16>, tensor<1x1x8x64xf32>

    return %result_running_output : tensor<1x8x64x128xf16>

    // CHECK-DAG:   [[IN_OUT:%.+]] = const.Declare tensor<1x8x64x128xf16> = dense<0.000000e+00> : tensor<1x8x64x128xf16>
    // CHECK-DAG:   [[IN_MAX:%.+]] = const.Declare tensor<1x1x8x64xf16> = dense<0xFC00> : tensor<1x1x8x64xf16>
    // CHECK-DAG:   [[IN_SUM:%.+]] = const.Declare tensor<1x1x8x64xf32> = dense<0.000000e+00> : tensor<1x1x8x64xf32>

    // CHECK-DAG:   [[IN_MAX_RESHAPED:%.+]] = VPU.AffineReshape([[IN_MAX]]) {dim_mapping = {{\[\[}}0], [0], [1], [2, 3]], shape_value = [1, 8, 64, 1]} : tensor<1x1x8x64xf16> -> tensor<1x8x64x1xf16>
    // CHECK-DAG:   [[IN_SUM_RESHAPED:%.+]] = VPU.AffineReshape([[IN_SUM]]) {dim_mapping = {{\[\[}}0], [0], [1], [2, 3]], shape_value = [1, 8, 64, 1]} : tensor<1x1x8x64xf32> -> tensor<1x8x64x1xf32>

    // CHECK-DAG:   [[AUX_BUF:%.+]] = VPU.Empty : tensor<1x2x64x32xf16>
    // CHECK-DAG:   [[DPU_DESCRIPTORS_BUF:%.+]] = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    // CHECK-DAG:   [[WEIGHTS_TABLE_0:%.+]] = const.Declare tensor<1x1x32x4xsi32> = dense
    // CHECK-DAG:   [[WEIGHTS_TABLE_1:%.+]] = const.Declare tensor<1x1x128x4xsi32> = dense

    // CHECK:           [[RES_OUT:%[^, ]+]], [[RES_MAX:%[^, ]+]], [[RES_SUM:%[^, ]+]] =
    // CHECK-SAME:              VPU.FlashSDPA([[QUERY]], [[KEY]], [[VALUE]], [[AUX_BUF]],
    // CHECK-SAME:                            [[DPU_DESCRIPTORS_BUF]], [[WEIGHTS_TABLE_0]], [[WEIGHTS_TABLE_1]],
    // CHECK-SAME:                            [[IN_OUT]], [[IN_MAX_RESHAPED]], [[IN_SUM_RESHAPED]]) {
    // CHECK-SAME:                      is_head = true,
    // CHECK-SAME:                      is_tail = true,
    // CHECK-SAME:                      source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:                  -> tensor<1x8x64x128xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x1xf32>

    // CHECK-DAG:   [[RESHAPED_RES_MAX:%.+]] = VPU.AffineReshape([[RES_MAX]]) {dim_mapping = {{\[\[}}0, 1], [2], [3], [3]], shape_value = [1, 1, 8, 64]} : tensor<1x8x64x1xf16> -> tensor<1x1x8x64xf16>
    // CHECK-DAG:   [[RESHAPED_RES_SUM:%.+]] = VPU.AffineReshape([[RES_SUM]]) {dim_mapping = {{\[\[}}0, 1], [2], [3], [3]], shape_value = [1, 1, 8, 64]} : tensor<1x8x64x1xf32> -> tensor<1x1x8x64xf32>

    // CHECK:   return [[RES_OUT]]
}

// -----

// CHECK-LABEL: @LogSoftmaxPeak
// CHECK-SAME: [[INPUT:%[^:]+]]: tensor<1x1x151x7056xf16>
func.func @LogSoftmaxPeak(%arg0: tensor<1x1x151x7056xf16>) -> (tensor<1x1x151x1xf32>, tensor<1x1x151x1xsi64>) {
    %output, %topKOutput = IE.LogSoftmaxPeak(%arg0) {axisInd = 3 : i64, dstElemType = f32, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x1xf32>, tensor<1x1x151x1xsi64>
    return %output, %topKOutput : tensor<1x1x151x1xf32>, tensor<1x1x151x1xsi64>

    // CHECK: [[OUTPUT:%.+]], [[TOPK_OUTPUT:%.+]] = VPU.LogSoftmaxPeak([[INPUT]]) {axisInd = 3 : i64, dstElemType = f32, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x1xf32>, tensor<1x1x151x1xsi64>
    // CHECK: return [[OUTPUT]], [[TOPK_OUTPUT]] : tensor<1x1x151x1xf32>, tensor<1x1x151x1xsi64>
}

// -----

// CHECK-LABEL: @SeqLenParamLSTMSequence
// CHECK-SAME: [[ARG0:%[^, ]+]]: tensor<2x2x25x512xf16>,
// CHECK-SAME: [[ARG1:%[^, ]+]]: tensor<2x2x1x128xf16>,
// CHECK-SAME: [[ARG2:%[^, ]+]]: tensor<2x2x1x128xf16>
// CHECK-SAME: [[ARG3:%[^, ]+]]: tensor<2x1x1x1xsi32>
func.func @SeqLenParamLSTMSequence (%arg0: tensor<2x2x25x512xf16>, %arg1: tensor<2x2x1x128xf16>, %arg2: tensor<2x2x1x128xf16>, %arg3: tensor<2x1x1x1xsi32>) -> (tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>) {
    %cst = const.Declare tensor<2x4x128x128xf16> = dense<0.000000e+00> : tensor<2x512x128xf32>, [#const.Reshape<[2, 4, 128, 128]>, #const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x2x4x128xf16> = dense<0.000000e+00> : tensor<2x512xf32>, [#const.Reshape<[1, 2, 4, 128]>, #const.CastElemType<f16>]
    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %arg3, %cst, %cst_0) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1>} : tensor<2x2x25x512xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>, tensor<2x1x1x1xsi32>, tensor<2x4x128x128xf16>, tensor<1x2x4x128xf16> -> tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>
    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<2x4x128x128xf16> = dense<0.000000e+00> : tensor<2x512x128xf32>, [#const.Reshape<[2, 4, 128, 128]>, #const.CastElemType<f16>]
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x2x4x128xf16> = dense<0.000000e+00> : tensor<2x512xf32>, [#const.Reshape<[1, 2, 4, 128]>, #const.CastElemType<f16>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x2x2432xsi32> = dense<0> : tensor<1x1x2x2432xsi32>
    // CHECK: [[OUT_HV:%.+]], [[OUT_HS:%.+]], [[OUT_CS:%.+]] = VPU.LSTMSequence([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]], [[CST]], [[CST_0]], [[CST_1]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>, useDpu = true} : tensor<2x2x25x512xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>, tensor<2x1x1x1xsi32>, tensor<2x4x128x128xf16>, tensor<1x2x4x128xf16>, tensor<1x1x2x2432xsi32> -> tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>
    // CHECK: return [[OUT_HV]], [[OUT_HS]], [[OUT_CS]] : tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>
}
