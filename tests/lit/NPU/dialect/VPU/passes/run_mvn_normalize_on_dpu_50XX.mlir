//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-mvn-normalize-on-dpu %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReplaceMVN1NormalizeWithMaxPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf32>
func.func @ReplaceMVN1NormalizeWithMaxPool(%arg0: tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32> {
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x512x64x64xf32> -> tensor<1x64x512x64xf32, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs(%0 : tensor<1x64x512x64xf32, {order = #NHWC}>) -> tensor<1x512x64x64xf32, {order = #NHWC}>
    %2 = VPU.Convert(%1) {dstElemType = f16} : tensor<1x512x64x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf16, {order = #NHWC}>
    %3 = VPU.ShapeCast {shape = [1, 512, 4096, 1]} inputs(%2 : tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    %4 = VPU.MVN1SumOp(%3) {across_channels = false, normalize_variance = true, output_height = 3 : i64} : tensor<1x512x4096x1xf16, {order = #NHWC}> -> tensor<1x512x3x2xf32, {order = #NHWC}>
    %5 = VPU.MVN1MeanVar(%4) {across_channels = false, eps = 9.9999999999999995E-7 : f64, internal_reshape = [1, 32, 16, 4096], normalize_variance = true, orig_shape = [1, 512, 64, 64], output_type = f16} : tensor<1x512x3x2xf32, {order = #NHWC}> -> tensor<1x512x1x2xf16, {order = #NHWC}>
    %6 = VPU.MVN1Normalize(%3, %5) {across_channels = false, normalize_variance = true} : tensor<1x512x4096x1xf16, {order = #NHWC}>, tensor<1x512x1x2xf16, {order = #NHWC}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    %7 = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs(%6 : tensor<1x512x4096x1xf16, {order = #NHWC}>) -> tensor<1x512x64x64xf16, {order = #NHWC}>
    %8 = VPU.Convert(%7) {dstElemType = f32} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x64x64xf32, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 64, 512, 64]} inputs(%8 : tensor<1x512x64x64xf32, {order = #NHWC}>) -> tensor<1x64x512x64xf32, {order = #NHWC}>
    %10 = VPU.PermuteCast(%9) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x512x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf32>
    return %10 : tensor<1x512x64x64xf32>

    // CHECK:        [[PERMUTECAST_1:%.+]] = VPU.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x512x64x64xf32> -> tensor<1x64x512x64xf32, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_1:%.+]] = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs([[PERMUTECAST_1]] : tensor<1x64x512x64xf32, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x512x64x64xf32, {order = #NHWC}>

    // CHECK:        [[CONVERT_1:%.+]] = VPU.Convert([[SHAPECAST_1]]) {dstElemType = f16}
    // CHECK-SAME:       : tensor<1x512x64x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_2:%.+]] = VPU.ShapeCast {shape = [1, 512, 4096, 1]} inputs([[CONVERT_1]] : tensor<1x512x64x64xf16, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x512x4096x1xf16, {order = #NHWC}>

    // CHECK:        [[SUM:%.+]] = VPU.MVN1SumOp([[SHAPECAST_2]]) {across_channels = false, normalize_variance = true, output_height = 3 : i64}
    // CHECK-SAME:       : tensor<1x512x4096x1xf16, {order = #NHWC}> -> tensor<1x512x3x2xf32, {order = #NHWC}>

    // CHECK:        [[MEANVAR:%.+]] = VPU.MVN1MeanVar([[SUM]])
    // CHECK-SAME:       {across_channels = false, eps = 9.9999999999999995E-7 : f64, internal_reshape = [1, 32, 16, 4096], normalize_variance = true, orig_shape = [1, 512, 64, 64], output_type = f16}
    // CHECK-SAME:       : tensor<1x512x3x2xf32, {order = #NHWC}> -> tensor<1x512x1x2xf16, {order = #NHWC}>

    // CHECK:        [[SLICE_MEAN:%.+]] = VPU.Slice [[MEANVAR]] [0, 0, 0, 0] [1, 512, 1, 1]
    // CHECK-SAME:       : tensor<1x512x1x2xf16, {order = #NHWC}> to tensor<1x512x1x1xf16, {order = #NHWC}>

    // CHECK:        [[SLICE_SCALE:%.+]] = VPU.Slice [[MEANVAR]] [0, 0, 0, 1] [1, 512, 1, 1]
    // CHECK-SAME:       : tensor<1x512x1x2xf16, {order = #NHWC}> to tensor<1x512x1x1xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_3:%.+]] = VPU.ShapeCast {shape = [512, 1, 1, 1]} inputs([[SLICE_SCALE]] : tensor<1x512x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<512x1x1x1xf16, {order = #NHWC}>

    // CHECK:        [[CONVERT_2:%.+]] = VPU.Convert([[SHAPECAST_3]]) {dstElemType = f32}
    // CHECK-SAME:       : tensor<512x1x1x1xf16, {order = #NHWC}> -> tensor<512x1x1x1xf32, {order = #NHWC}>

    // CHECK-DAG:    [[CST_NEG:%.+]] = const.Declare tensor<1x512x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x512x1x1xf16>

    // CHECK:        [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE_MEAN]], [[CST_NEG]])
    // CHECK-SAME:       {op_type = #VPU.eltwise_type<MULTIPLY>

    // CHECK:        [[CONVERT_3:%.+]] = VPU.Convert([[ELTWISE]]) {dstElemType = f32}
    // CHECK-SAME:       : tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf32, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_4:%.+]] = VPU.ShapeCast {shape = [512, 1, 1, 1]} inputs([[CONVERT_3]] : tensor<1x512x1x1xf32, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<512x1x1x1xf32, {order = #NHWC}>

    // CHECK-DAG:    [[CST_ZEROS:%.+]] = const.Declare tensor<512x1x1x2xf32, {order = #NHWC}> = dense<0.000000e+00> : tensor<512x1x1x2xf32>

    // CHECK:        [[CONCAT:%.+]] = VPU.Concat([[CST_ZEROS]], [[CONVERT_2]], [[SHAPECAST_4]])
    // CHECK-SAME:       {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 3]]}

    // CHECK:        [[REINTERPRET:%.+]] = Core.ReinterpretCast([[CONCAT]])
    // CHECK-SAME:       : tensor<512x1x1x4xf32, {order = #NHWC}> -> tensor<512x1x1x4xsi32, {order = #NHWC}>

    // CHECK:        [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SHAPECAST_2]], [[REINTERPRET]] )
    // CHECK-SAME:       -> tensor<1x512x4096x1xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_5:%.+]] = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs([[MAXPOOL]] : tensor<1x512x4096x1xf16, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x512x64x64xf16, {order = #NHWC}>

    // CHECK:        [[CONVERT_4:%.+]] = VPU.Convert([[SHAPECAST_5]]) {dstElemType = f32}
    // CHECK-SAME:       : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x64x64xf32, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_6:%.+]] = VPU.ShapeCast {shape = [1, 64, 512, 64]} inputs([[CONVERT_4]] : tensor<1x512x64x64xf32, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x64x512x64xf32, {order = #NHWC}>

    // CHECK:        [[PERMUTECAST_2:%.+]] = VPU.PermuteCast([[SHAPECAST_6]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x64x512x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf32>

    // CHECK:        return [[PERMUTECAST_2]] : tensor<1x512x64x64xf32>
}
