//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-reduce-mean-square --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FuseReduceMeanSquare
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf32>)
func.func @FuseReduceMeanSquare(%arg0: tensor<1x32x32x96xf32>) -> tensor<1x32x32x1xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf32>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
    %2 = IE.Sqrt(%1) : tensor<1x32x32x1xf32> -> tensor<1x32x32x1xf32>
    return %2 : tensor<1x32x32x1xf32>

    // CHECK: [[ReduceMeanSquare:%.+]] = IE.ReduceMeanSquare([[ARG0]]) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
    // CHECK: return [[ReduceMeanSquare]] : tensor<1x32x32x1xf32>
}

// -----

// CHECK-LABEL: @FuseReduceMeanSquareWithAdd
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf16>)
func.func @FuseReduceMeanSquareWithAdd(%arg0: tensor<1x32x32x96xf16>) -> tensor<1x32x32x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32> 
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x1xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x1xf16>
    %3 = IE.Sqrt(%2) : tensor<1x32x32x1xf16> -> tensor<1x32x32x1xf16>
    return %3 : tensor<1x32x32x1xf16>

    // CHECK: [[ReduceMeanSquare:%.+]] = IE.ReduceMeanSquare([[ARG0]]) {axes_value = [3], epsilon = 3.000000e+00 : f64, keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    // CHECK: return [[ReduceMeanSquare]] : tensor<1x32x32x1xf16>
}

// -----

// CHECK-LABEL: @NoFuseReduceMeanSquareEpsilonNonInnermost
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf16>)
func.func @NoFuseReduceMeanSquareEpsilonNonInnermost(%arg0: tensor<1x32x32x96xf16>) -> tensor<1x1x32x96xf16> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32> 
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
    %1 = IE.ReduceMean(%0) {axes_value = [1], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x1x32x96xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x1x32x96xf16>
    %3 = IE.Sqrt(%2) : tensor<1x1x32x96xf16> -> tensor<1x1x32x96xf16>
    return %3 : tensor<1x1x32x96xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[POWER:%.+]] = IE.Power([[ARG0]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
    // CHECK: [[REDUCE:%.+]] = IE.ReduceMean([[POWER]]) {axes_value = [1], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x1x32x96xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[REDUCE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x1x32x96xf16>
    // CHECK: [[SQRT:%.+]] = IE.Sqrt([[ADD]]) : tensor<1x1x32x96xf16> -> tensor<1x1x32x96xf16>
    // CHECK: return [[SQRT]] : tensor<1x1x32x96xf16>
}
