//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-divide-to-multiply --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f16, 0.01013327205882353>

// CHECK-LABEL: @DoNotConvertNonConstDivide
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @DoNotConvertNonConstDivide(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x1x1x1x!qElemType> = dense<2> : tensor<1x1x1x1xui8>, [#const.CastElemType<!qElemType>]
    %nonCst = IE.Dequantize(%divisor) {dstElemType = f16} : tensor<1x1x1x1x!qElemType> -> tensor<1x1x1x1xf16>
    %0 = IE.Divide(%arg, %nonCst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1x!qElemType>
    // CHECK: [[NON_CONST:%.+]] = IE.Dequantize([[CONST]])
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[ARG]], [[NON_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[DIVIDE]]
}

// -----

// CHECK-LABEL: @DoNotConvertIntegerDivide
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xsi32>) -> tensor<1x12x512x512xsi32>
func.func @DoNotConvertIntegerDivide(%arg: tensor<1x12x512x512xsi32>) -> tensor<1x12x512x512xsi32> {
    %divisor = const.Declare tensor<1x1x1x1xsi32> = dense<2> : tensor<1x1x1x1xsi32>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xsi32>, tensor<1x1x1x1xsi32> -> tensor<1x12x512x512xsi32>
    return %0 : tensor<1x12x512x512xsi32>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xsi32>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[DIVIDE]]
}

// -----

// CHECK-LABEL: @DoNotConvertConstDividend
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @DoNotConvertConstDividend(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Divide(%cst, %arg) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x1x1xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CONST]], [[ARG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[DIVIDE]]
}

// -----

// CHECK-LABEL: @ConvertConstDivisor
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @ConvertConstDivisor(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf16> {{.*}} [#const.ScalarMultInverse]
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @ConvertConstDivisor_NonScalar
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @ConvertConstDivisor_NonScalar(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x12x512x512xf16> {{.*}} [#const.ScalarMultInverse]
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOps
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x512x512xf16>, [[ARG1:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @ConvertMultipleDivideOps(%arg0: tensor<1x12x512x512xf16>, %arg1: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg0, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Divide(%arg1, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK:     [[CONST:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>, [#const.ScalarMultInverse]
    // CHECK-DAG: [[MULTIPLY0:%.+]] = IE.Multiply([[ARG0]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-DAG: [[MULTIPLY1:%.+]] = IE.Multiply([[ARG1]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:     return [[MULTIPLY0]], [[MULTIPLY1]]
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOps_NotSecondInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x512x512xf16>, [[ARG1:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @ConvertMultipleDivideOps_NotSecondInput(%arg0: tensor<1x12x512x512xf16>, %arg1: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg0, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Divide(%divisor, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK: [[CONST0:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>, [#const.ScalarMultInverse]
    // CHECK: [[CONST1:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CONST1]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[MULTIPLY]], [[DIVIDE]]
}

// -----

// CHECK-LABEL: @ConvertNotOnlyDivideUsers
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x512x512xf16>, [[ARG1:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @ConvertNotOnlyDivideUsers(%arg0: tensor<1x12x512x512xf16>, %arg1: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %cst = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Add(%arg1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK: [[CONST0:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>, [#const.ScalarMultInverse]
    // CHECK: [[CONST1:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: [[ADD:%.+]] = IE.Add([[ARG1]], [[CONST1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[MULTIPLY]], [[ADD]]
}

// -----

// CHECK-LABEL: @ConvertDivideWithConstQuantizedDivisor
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16>
func.func @ConvertDivideWithConstQuantizedDivisor(%arg0: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16> {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1 : tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.ScalarMultInverse]
    // CHECK-DAG:       [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.608150e-03> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.936340e-02> : tensor<1x1x1x1xf16>
    // CHECK:           [[FAKE_QUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[ARG]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOpsWithConstQuantizedDivisor
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithConstQuantizedDivisor(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.ScalarMultInverse]
    // CHECK-DAG:       [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.608150e-03> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.936340e-02> : tensor<1x1x1x1xf16>
    // CHECK:           [[FAKE_QUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[MULTIPLY0:%.+]] = IE.Multiply([[ARG0]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           [[MULTIPLY1:%.+]] = IE.Multiply([[ARG1]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[MULTIPLY0]], [[MULTIPLY1]]
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOpsWithConstQuantizedDivisor_NotOnlyDivideUsers
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithConstQuantizedDivisor_NotOnlyDivideUsers(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>

    %1 = IE.Add(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CONST0:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.ScalarMultInverse]
    // CHECK-DAG:       [[CONST1:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]

    // CHECK-DAG:       [[IN_LOW1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:       [[IN_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:       [[OUT_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK-DAG:       [[IN_LOW0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[IN_HIGH0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.608150e-03> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[OUT_HIGH0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.936340e-02> : tensor<1x1x1x1xf16>

    // CHECK:           [[FAKE_QUANTIZE0:%.+]] = IE.FakeQuantize([[CONST0]], [[IN_LOW0]], [[IN_HIGH0]], [[IN_LOW0]], [[OUT_HIGH0]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[FAKE_QUANTIZE1:%.+]] = IE.FakeQuantize([[CONST1]], [[IN_LOW1]], [[IN_HIGH1]], [[IN_LOW1]], [[OUT_HIGH1]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}

    // CHECK:           [[ADD:%.+]] = IE.Add([[ARG0]], [[FAKE_QUANTIZE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[ARG1]], [[FAKE_QUANTIZE0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[ADD]], [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOpsWithQuantizedDivisor_NotSecondInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithQuantizedDivisor_NotSecondInput(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%0, %arg0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x1x1xf16>, tensor<1x3x768x1152xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CONST0:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.ScalarMultInverse]
    // CHECK-DAG:       [[CONST1:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:           [[IN_LOW1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:           [[IN_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:           [[OUT_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:           [[IN_LOW0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:           [[IN_HIGH0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.608150e-03> : tensor<1x1x1x1xf16>
    // CHECK:           [[OUT_HIGH0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.936340e-02> : tensor<1x1x1x1xf16>

    // CHECK:           [[FAKE_QUANTIZE0:%.+]] = IE.FakeQuantize([[CONST0]], [[IN_LOW0]], [[IN_HIGH0]], [[IN_LOW0]], [[OUT_HIGH0]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[FAKE_QUANTIZE1:%.+]] = IE.FakeQuantize([[CONST1]], [[IN_LOW1]], [[IN_HIGH1]], [[IN_LOW1]], [[OUT_HIGH1]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}

    // CHECK:           [[DIVIDE:%.+]] = IE.Divide([[FAKE_QUANTIZE1]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[ARG1]], [[FAKE_QUANTIZE0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[DIVIDE]], [[MULTIPLY]]
}
