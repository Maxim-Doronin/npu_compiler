//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FoldReduceL1
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceL1(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceL1(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceL1
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceL2
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceL2(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceL2
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceLogicalAnd
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceLogicalAnd(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceLogicalAnd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceLogicalAnd
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceLogicalOr
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceLogicalOr(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceLogicalOr(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceLogicalOr
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceMax
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMax(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMax
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceMean
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMean(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMean
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @DoNotFoldPaddedReduceMean
func.func @DoNotFoldPaddedReduceMean(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x16x4x2xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x16x4x2xf16>
    return %0 : tensor<1x16x4x2xf16>

    // CHECK:   [[REDUCE:%.+]] = IE.ReduceMean
    // CHECK:   return [[REDUCE]]
}

// -----

// CHECK-LABEL: @FoldReduceMin
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMin(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMin(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMin
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceProd
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceProd(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceProd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceProd
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceSum
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceSum(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceSum
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @DoNotFoldPaddedReduceSum
func.func @DoNotFoldPaddedReduceSum(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x16x4x2xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x16x4x2xf16>
    return %0 : tensor<1x16x4x2xf16>

    // CHECK:   [[REDUCE:%.+]] = IE.ReduceSum
    // CHECK:   return [[REDUCE]]
}

// -----

// CHECK-LABEL: @ConvertToAttrReduceL1
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceL1(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceL1(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_L1:%.+]] = IE.ReduceL1([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_L1]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceL2
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceL2(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_L2:%.+]] = IE.ReduceL2([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_L2]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceLogicalAnd
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceLogicalAnd(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceLogicalAnd(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_LOGICAL_AND:%.+]] = IE.ReduceLogicalAnd([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_LOGICAL_AND]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceLogicalOr
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceLogicalOr(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceLogicalOr(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_LOGICAL_OR:%.+]] = IE.ReduceLogicalOr([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_LOGICAL_OR]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceMax
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceMax(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_MAX:%.+]] = IE.ReduceMax([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_MAX]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceMean
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceMean(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_MEAN:%.+]] = IE.ReduceMean([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_MEAN]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceMin
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceMin(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceMin(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_MIN:%.+]] = IE.ReduceMin([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_MIN]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceProd
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceProd(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceProd(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_PROD:%.+]] = IE.ReduceProd([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_PROD]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceSum
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x4x2xf16>)
func.func @ConvertToAttrReduceSum(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_SUM:%.+]] = IE.ReduceSum([[ARG_0]]) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_SUM]] : tensor<1x1x4x2xf16>

}
