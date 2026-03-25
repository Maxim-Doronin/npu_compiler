//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-mem-permute-processing-rewriters="se-ops-enabled=true rewriter=swap-operations-set"  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapReLUwithInterpolate
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<16x512x1x1xf16>, [[ARG1:%.+]]: tensor<1024x512x1x1xf16>)
func.func @SwapReLUwithInterpolate(%arg0: tensor<16x512x1x1xf16>, %arg1: tensor<1024x512x1x1xf16>) -> tensor<1x16x2048x2xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<16x512x1x1xf16>, tensor<1024x512x1x1xf16> -> tensor<16x1024x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16, 1024, 1]} : tensor<16x1024x1x1xf16> -> tensor<1x16x1024x1xf16>
    %2 = IE.Interpolate(%1) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [2048, 2]
         } : tensor<1x16x1024x1xf16> -> tensor<1x16x2048x2xf16>
    %3 = IE.ReLU(%2) : tensor<1x16x2048x2xf16> -> tensor<1x16x2048x2xf16>

    return %3 : tensor<1x16x2048x2xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<16x512x1x1xf16>, tensor<1024x512x1x1xf16> -> tensor<16x1024x1x1xf16>
    // CHECK: IE.ReLU
    // CHECK-SAME: tensor<16x1024x1x1xf16> -> tensor<16x1024x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<16x1024x1x1xf16> -> tensor<1x16x1024x1xf16>
    // CHECK: IE.Interpolate
    // CHECK-SAME: tensor<1x16x1024x1xf16> -> tensor<1x16x2048x2xf16>

}
