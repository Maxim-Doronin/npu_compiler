//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-gather %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertGatherToSliceAxis0
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<18x8x72x64xf16>)
func.func @ConvertGatherToSliceAxis0(%arg0: tensor<18x8x72x64xf16>) -> tensor<8x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<9> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<8x72x64xf16>

    return %0 : tensor<8x72x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [9, 0, 0, 0] [1, 8, 72, 64] : tensor<18x8x72x64xf16> to tensor<1x8x72x64xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SLICE]]) {shape_value = [8, 72, 64]} : tensor<1x8x72x64xf16> -> tensor<8x72x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// -----

// CHECK-LABEL: @ConvertGatherToSliceAxis1
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<18x8x72x64xf16>)
func.func @ConvertGatherToSliceAxis1(%arg0: tensor<18x8x72x64xf16>) -> tensor<18x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<3> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<18x72x64xf16>

    return %0 : tensor<18x72x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 3, 0, 0] [18, 1, 72, 64] : tensor<18x8x72x64xf16> to tensor<18x1x72x64xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SLICE]]) {shape_value = [18, 72, 64]} : tensor<18x1x72x64xf16> -> tensor<18x72x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// CHECK-LABEL: @ConvertGatherToSlicewith3DShape
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<8x72x64xf16>)
func.func @ConvertGatherToSlicewith3DShape(%arg0: tensor<8x72x64xf16>) -> tensor<8x64xf16> {
    %cst = const.Declare tensor<si32> = dense<8> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<8x72x64xf16>, tensor<si32> -> tensor<8x64xf16>

    return %0 : tensor<8x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 8, 0] [8, 1, 64] : tensor<8x72x64xf16> to tensor<8x1x64xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SLICE]]) {shape_value = [8, 64]} : tensor<8x1x64xf16> -> tensor<8x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// CHECK-LABEL: @CannotConvertGatherToSlice
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1xf32>, [[ARG_1:%[^:]+]]: tensor<1x8x16x16xf16>)
func.func @CannotConvertGatherToSlice(%arg0: tensor<1xf32>, %arg1: tensor<1x8x16x16xf16>) -> tensor<1x1x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = si32} : tensor<1xf32> -> tensor<1xsi32>
    %1 = IE.Gather(%arg1, %0) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x8x16x16xf16>, tensor<1xsi32> -> tensor<1x1x16x16xf16>

    return %1 : tensor<1x1x16x16xf16>

    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[ARG_0]]) {dstElemType = si32} : tensor<1xf32> -> tensor<1xsi32>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[ARG_1]], [[CONVERT]]) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x8x16x16xf16>, tensor<1xsi32> -> tensor<1x1x16x16xf16>
    // CHECK:       return [[GATHER]] : tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL: @ConvertGatherToReverseWithReverseContiguousIndices
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<16x224x224x3xf16>
func.func @ConvertGatherToReverseWithReverseContiguousIndices(%arg0: tensor<16x224x224x3xf16>) -> tensor<16x224x224x3xf16> {
    %cst = const.Declare tensor<3xsi32> = dense<[2, 1, 0]> : tensor<3xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 3 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<16x224x224x3xf16>, tensor<3xsi32> -> tensor<16x224x224x3xf16>

    return %0 : tensor<16x224x224x3xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[INPUT]]) {axis_value = [3], mode = #IE.reverse_mode<INDEX>} : tensor<16x224x224x3xf16> -> tensor<16x224x224x3xf16>
    // CHECK:       return [[REVERSE]] : tensor<16x224x224x3xf16>
}

// -----

// CHECK-LABEL: @NotConvertGatherToReverseWithIndicesFirst
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x640x640xf16>
func.func @NotConvertGatherToReverseWithIndicesFirst(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x3x640x640xf16> {
    %cst = const.Declare tensor<3xsi32> = dense<[2, 1, 0]> : tensor<3xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1x3x640x640xf16>, tensor<3xsi32> -> tensor<1x3x640x640xf16>

    return %0 : tensor<1x3x640x640xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<3xsi32> = dense<[2, 1, 0]> : tensor<3xsi32>
    // CHECK:       [[Gather:%.+]] = IE.Gather([[INPUT]], [[CST]]) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1x3x640x640xf16>, tensor<3xsi32> -> tensor<1x3x640x640xf16>
    // CHECK:       return [[Gather]] : tensor<1x3x640x640xf16>
}
