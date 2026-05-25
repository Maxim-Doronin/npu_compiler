//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --fuse-convert --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseD2sConvert
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x20x40xf16, {order = #NHWC}>
func.func @FuseD2sConvert(%input: tensor<1x16x20x40xf16, {order = #NHWC}>) -> tensor<1x40x80x4xf32> {
    %0 = VPU.DepthToSpace(%input) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
         : tensor<1x16x20x40xf16, {order = #NHWC}> -> tensor<1x4x40x80xf16, {order = #NHWC}>
    %1 = VPU.Convert(%0) {dstElemType = f32} : tensor<1x4x40x80xf16, {order = #NHWC}> -> tensor<1x4x40x80xf32, {order = #NHWC}>
    %2 = VPU.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x4x40x80xf32, {order = #NHWC}> -> tensor<1x40x80x4xf32>
    return %2 : tensor<1x40x80x4xf32>

    // CHECK:       [[D2S_CVT:%.+]] = VPU.DepthToSpace([[INPUT]])
    // CHECK-SAME:                       {block_size = 2 : i64, dstElemType = f32, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
    // CHECK-SAME:                       : tensor<1x16x20x40xf16, {order = #NHWC}> -> tensor<1x4x40x80xf32, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]]  = VPU.PermuteCast([[D2S_CVT]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:                       : tensor<1x4x40x80xf32, {order = #NHWC}> -> tensor<1x40x80x4xf32>

    // CHECK:       return [[OUTPUT]] : tensor<1x40x80x4xf32>
}
