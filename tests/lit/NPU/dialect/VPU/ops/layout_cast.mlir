//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @Fold
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x8x64xf32>
func.func @Fold(%arg0: tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NCHW} : tensor<1x4x8x64xf32> -> tensor<1x4x8x64xf32>
    return %0 : tensor<1x4x8x64xf32>

    // CHECK-NOT:  VPU.LayoutCast
    // CHECK:      return [[INPUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x2x3x4xf32> {
    %0 = const.Declare tensor<1x2x3x4xf32, {order = #NHWC}> = dense<5.0> : tensor<1x2x3x4xf32>, [#const.Reorder<#NHWC>]
    %1 = VPU.LayoutCast(%0) {dst_order = #NCHW} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>
    return %1 : tensor<1x2x3x4xf32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x2x3x4xf32> =
    // CHECK-SAME:      dense<5.000000e+00> : tensor<1x2x3x4xf32>, [#const.Reorder<#NHWC>, #const.LayoutCast<#NCHW>]
    // CHECK-NOT:   VPU.LayoutCast
    // CHECK:       return [[CST]]
}
