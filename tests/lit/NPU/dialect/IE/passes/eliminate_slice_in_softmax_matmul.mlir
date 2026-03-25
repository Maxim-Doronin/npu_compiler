//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --eliminate-slice-in-softmax-matmul %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @EliminateSoftmaxSliceMatMul
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1024x1025xf16>, [[ARG1:%.+]]: tensor<1x64x64x1024xf16>)
func.func @EliminateSoftmaxSliceMatMul(%arg0: tensor<1x64x1024x1025xf16>, %arg1: tensor<1x64x64x1024xf16>) -> tensor<1x64x1024x64xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 3} : tensor<1x64x1024x1025xf16> -> tensor<1x64x1024x1025xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 64, 1024, 1024] : tensor<1x64x1024x1025xf16> to tensor<1x64x1024x1024xf16>
    %2 = IE.MatMul(%1, %arg1) {transpose_b} : tensor<1x64x1024x1024xf16>, tensor<1x64x64x1024xf16> -> tensor<1x64x1024x64xf16>
    return %2 : tensor<1x64x1024x64xf16>

    // Pattern: Softmax(1025) -> Slice[0:1024] -> MatMul
    // Optimization: Pad Softmax to 1040 (aligned), pad RHS to 1040, remove Slice

    // CHECK-DAG:   [[NEGINF:%.+]] = const.Declare tensor<1x64x1024x15xf16> = dense<0xFC00> : tensor<1x64x1024x15xf16>
    // CHECK:       [[SOFTMAX_INPUT:%.+]] = IE.Concat([[ARG0]], [[NEGINF]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:      : tensor<1x64x1024x1025xf16>, tensor<1x64x1024x15xf16> -> tensor<1x64x1024x1040xf16>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[SOFTMAX_INPUT]]) {axisInd = 3 : i64}
    // CHECK-SAME:      : tensor<1x64x1024x1040xf16> -> tensor<1x64x1024x1040xf16>
    // CHECK-DAG:   [[ZERO:%.+]] = const.Declare tensor<1x64x64x16xf16> = dense<0.000000e+00> : tensor<1x64x64x16xf16>
    // CHECK:       [[RHS_PADDED:%.+]] = IE.Concat([[ARG1]], [[ZERO]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:      : tensor<1x64x64x1024xf16>, tensor<1x64x64x16xf16> -> tensor<1x64x64x1040xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[SOFTMAX]], [[RHS_PADDED]]) {transpose_b}
    // CHECK-SAME:      : tensor<1x64x1024x1040xf16>, tensor<1x64x64x1040xf16> -> tensor<1x64x1024x64xf16>
    // CHECK:       return [[MATMUL]] : tensor<1x64x1024x64xf16>
}

// -----

// CHECK-LABEL: @SkipSoftmaxAlreadyAligned
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1024x1024xf16>, [[ARG1:%.+]]: tensor<1x64x64x1020xf16>)
func.func @SkipSoftmaxAlreadyAligned(%arg0: tensor<1x64x1024x1024xf16>, %arg1: tensor<1x64x64x1020xf16>) -> tensor<1x64x1024x64xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 3} : tensor<1x64x1024x1024xf16> -> tensor<1x64x1024x1024xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 64, 1024, 1020] : tensor<1x64x1024x1024xf16> to tensor<1x64x1024x1020xf16>
    %2 = IE.MatMul(%1, %arg1) {transpose_b} : tensor<1x64x1024x1020xf16>, tensor<1x64x64x1020xf16> -> tensor<1x64x1024x64xf16>
    return %2 : tensor<1x64x1024x64xf16>

    // Softmax dimension 1024 is already aligned to 32, skip optimization
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64}
    // CHECK-SAME:      : tensor<1x64x1024x1024xf16> -> tensor<1x64x1024x1024xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 0] [1, 64, 1024, 1020]
    // CHECK-SAME:      : tensor<1x64x1024x1024xf16> to tensor<1x64x1024x1020xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[SLICE]], [[ARG1]]) {transpose_b}
    // CHECK-SAME:      : tensor<1x64x1024x1020xf16>, tensor<1x64x64x1020xf16> -> tensor<1x64x1024x64xf16>
    // CHECK:       return [[MATMUL]] : tensor<1x64x1024x64xf16>
}

// -----

// CHECK-LABEL: @SkipNoSlice
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1024x1025xf16>, [[ARG1:%.+]]: tensor<1x64x64x1025xf16>)
func.func @SkipNoSlice(%arg0: tensor<1x64x1024x1025xf16>, %arg1: tensor<1x64x64x1025xf16>) -> tensor<1x64x1024x64xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 3} : tensor<1x64x1024x1025xf16> -> tensor<1x64x1024x1025xf16>
    %1 = IE.MatMul(%0, %arg1) {transpose_b} : tensor<1x64x1024x1025xf16>, tensor<1x64x64x1025xf16> -> tensor<1x64x1024x64xf16>
    return %1 : tensor<1x64x1024x64xf16>

    // No Slice operation, pattern doesn't match
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64}
    // CHECK-SAME:      : tensor<1x64x1024x1025xf16> -> tensor<1x64x1024x1025xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG1]]) {transpose_b}
    // CHECK-SAME:      : tensor<1x64x1024x1025xf16>, tensor<1x64x64x1025xf16> -> tensor<1x64x1024x64xf16>
    // CHECK:       return [[MATMUL]] : tensor<1x64x1024x64xf16>
}

// -----

// CHECK-LABEL: @UnrolledSDPACase
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x1024x1025xf32>, [[ARG1:%.+]]: tensor<1x1x64x1024xf32>)
func.func @UnrolledSDPACase(%arg0: tensor<1x1x1024x1025xf32>, %arg1: tensor<1x1x64x1024xf32>) -> tensor<1x1x1024x64xf32> {
    %0 = IE.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x1024x1025xf32> -> tensor<1x1x1024x1025xf32>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1, 1024, 1024] : tensor<1x1x1024x1025xf32> to tensor<1x1x1024x1024xf32>
    %2 = IE.MatMul(%1, %arg1) {transpose_b} : tensor<1x1x1024x1024xf32>, tensor<1x1x64x1024xf32> -> tensor<1x1x1024x64xf32>
    return %2 : tensor<1x1x1024x64xf32>

    // Typical unrolled SDPA pattern from batch dimension
    // Softmax(1025) -> Slice[0:1024] -> MatMul
    // Should be optimized to aligned dimensions

    // CHECK-DAG:   [[NEGINF:%.+]] = const.Declare tensor<1x1x1024x15xf32> = dense<0xFF800000> : tensor<1x1x1024x15xf32>
    // CHECK:       [[SOFTMAX_INPUT:%.+]] = IE.Concat([[ARG0]], [[NEGINF]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:      : tensor<1x1x1024x1025xf32>, tensor<1x1x1024x15xf32> -> tensor<1x1x1024x1040xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[SOFTMAX_INPUT]]) {axisInd = 3 : i64}
    // CHECK-SAME:      : tensor<1x1x1024x1040xf32> -> tensor<1x1x1024x1040xf32>
    // CHECK-DAG:   [[ZERO:%.+]] = const.Declare tensor<1x1x64x16xf32> = dense<0.000000e+00> : tensor<1x1x64x16xf32>
    // CHECK:       [[RHS_PADDED:%.+]] = IE.Concat([[ARG1]], [[ZERO]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:      : tensor<1x1x64x1024xf32>, tensor<1x1x64x16xf32> -> tensor<1x1x64x1040xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[SOFTMAX]], [[RHS_PADDED]]) {transpose_b}
    // CHECK-SAME:      : tensor<1x1x1024x1040xf32>, tensor<1x1x64x1040xf32> -> tensor<1x1x1024x64xf32>
    // CHECK:       return [[MATMUL]] : tensor<1x1x1024x64xf32>
}

// -----

// CHECK-LABEL: @SkipNonInnermostAxis
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1025x1024xf16>, [[ARG1:%.+]]: tensor<1x64x64x1024xf16>)
func.func @SkipNonInnermostAxis(%arg0: tensor<1x64x1025x1024xf16>, %arg1: tensor<1x64x64x1024xf16>) -> tensor<1x64x1024x64xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 2} : tensor<1x64x1025x1024xf16> -> tensor<1x64x1025x1024xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 64, 1024, 1024] : tensor<1x64x1025x1024xf16> to tensor<1x64x1024x1024xf16>
    %2 = IE.MatMul(%1, %arg1) {transpose_b} : tensor<1x64x1024x1024xf16>, tensor<1x64x64x1024xf16> -> tensor<1x64x1024x64xf16>
    return %2 : tensor<1x64x1024x64xf16>

    // Softmax axis=2 is not the innermost dimension (axis=3 is)
    // Pattern should not match
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 2 : i64}
    // CHECK-SAME:      : tensor<1x64x1025x1024xf16> -> tensor<1x64x1025x1024xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 0] [1, 64, 1024, 1024]
    // CHECK-SAME:      : tensor<1x64x1025x1024xf16> to tensor<1x64x1024x1024xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[SLICE]], [[ARG1]]) {transpose_b}
    // CHECK-SAME:      : tensor<1x64x1024x1024xf16>, tensor<1x64x64x1024xf16> -> tensor<1x64x1024x64xf16>
    // CHECK:       return [[MATMUL]] : tensor<1x64x1024x64xf16>
}

// -----

// CHECK-LABEL: @SkipShrinkMatMul
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1x1153xf32>, [[ARG1:%.+]]: tensor<1x8x1x64x1152xf32>)
func.func @SkipShrinkMatMul(%arg0: tensor<1x64x1x1153xf32>, %arg1: tensor<1x8x1x64x1152xf32>) -> tensor<1x64x1x64xf32> {
    %cst = const.Declare tensor<5xsi32> = dense<[1, 8, 8, 64, 1152]> : tensor<5xsi32>

    %0 = IE.SoftMax(%arg0) {axisInd = 3} : tensor<1x64x1x1153xf32> -> tensor<1x64x1x1153xf32>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 64, 1, 1152] : tensor<1x64x1x1153xf32> to tensor<1x64x1x1152xf32>

    %2 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x64x1152xf32>, tensor<5xsi32> -> tensor<1x8x8x64x1152xf32>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 64, 64, 1152]} : tensor<1x8x8x64x1152xf32> -> tensor<1x64x64x1152xf32>
    %4 = IE.MatMul(%1, %3) {transpose_b} : tensor<1x64x1x1152xf32>, tensor<1x64x64x1152xf32> -> tensor<1x64x1x64xf32>

    return %4 : tensor<1x64x1x64xf32>

    // This MatMul would benefit from shrinking groups
    // Padding RHS with zeros would break the shrink matmul pattern
    // Pattern should not be optimized
    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 8, 8, 64, 1152]> : tensor<5xsi32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64}
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 0] [1, 64, 1, 1152]
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[ARG1]], [[CST]])
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[SLICE]], [[RESHAPE]]) {transpose_b}
    // CHECK:       return [[MATMUL]] : tensor<1x64x1x64xf32>
}
