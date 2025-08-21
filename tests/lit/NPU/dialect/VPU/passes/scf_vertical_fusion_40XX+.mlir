//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --scf-vertical-fusion --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

IE.TileResource 3 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 960, 26)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 959)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0) -> (d0 + 6)>

// CHECK-LABEL: @MergeVFChain3Tiles
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x256x540x120xf16, {order = #NHWC}>)
func.func @MergeVFChain3Tiles(%arg0: tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_6 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_7 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_8 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_9 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_10 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_11 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_12 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_13 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs(%arg0 : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst, %cst_13) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst_12) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1, %cst_11) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst_10, %cst_9) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %cst_2, %cst_8) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %6 = VPU.NCE.Convolution(%5, %cst_3, %cst_7) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %cst_4, %cst_6) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %8 = VPU.NCE.DepthConvolution(%7, %cst_5, %cst_9) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 20]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%8 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>

    return %9: tensor<1x128x540x240xf16, {order = #NHWC}>

    //CHECK: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 26 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 960 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    //CHECK: [[CAST_INPUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {
    //CHECK:        [[OUTPUT_SIZE:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER]])
    //CHECK:        [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:        [[TEMP_VALUE1:%.+]] = affine.min #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:        [[PAD_LOW5:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:        [[TEMP_VALUE2:%.+]] = affine.min #[[$MAP4]]([[LOOP_ITER]], [[TEMP_VALUE0]])
    //CHECK:        [[PAD_HIGH5:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE2]]]
    //CHECK:        [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE0]])
    //CHECK:        [[TEMP_VALUE3:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE0]])
    //CHECK:        [[PAD_LOW4:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE3]]]
    //CHECK:        [[TEMP_VALUE4:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE0]], [[TEMP_VALUE2]])
    //CHECK:        [[PAD_HIGH4:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE4]]]
    //CHECK:        [[TEMP_VALUE5:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE2]])
    //CHECK:        [[TEMP_VALUE6:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE2]])
    //CHECK:        [[PAD_LOW3:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE6]]]
    //CHECK:        [[TEMP_VALUE7:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE2]], [[TEMP_VALUE5]])
    //CHECK:        [[PAD_HIGH3:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE7]]]
    //CHECK:        [[TEMP_VALUE8:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE5]])
    //CHECK:        [[TEMP_VALUE9:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE5]])
    //CHECK:        [[PAD_LOW2:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE9]]]
    //CHECK:        [[TEMP_VALUE10:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE5]], [[TEMP_VALUE8]])
    //CHECK:        [[PAD_HIGH2:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE10]]]
    //CHECK:        [[TEMP_VALUE11:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE8]])
    //CHECK:        [[TEMP_VALUE12:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE8]])
    //CHECK:        [[PAD_LOW1:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE12]]]
    //CHECK:        [[TEMP_VALUE13:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE8]], [[TEMP_VALUE11]])
    //CHECK:        [[PAD_HIGH1:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE13]]]
    //CHECK:        [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE11]])
    //CHECK:        [[TEMP_VALUE14:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE11]])
    //CHECK:        [[PAD_LOW0:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE14]]]
    //CHECK:        [[SLICE_SIZE:%.+]] = affine.apply #[[$MAP5]]([[OUTPUT_SIZE]])
    //CHECK:        [[TEMP_VALUE15:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE11]], [[SLICE_OFFSET]])
    //CHECK:        [[PAD_HIGH0:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE15]]]
    //CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SLICE_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {order = #NHWC}>

    //CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:        tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW3]]] high[0, 0, 1, [[PAD_HIGH3]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, 1, [[PAD_LOW4]]] high[0, 0, 1, [[PAD_HIGH4]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, 1, [[PAD_LOW5]]] high[0, 0, 1, [[PAD_HIGH5]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 540, [[OUTPUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:    scf.yield [[INSERT]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:    [[CAST:%.+]] = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs([[LOOP]]
    //CHECK:    return [[CAST]] : tensor<1x128x540x240xf16, {order = #NHWC}>
}

// -----

IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 960, 44)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 959)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0) -> (d0 + 6)>

// CHECK-LABEL: @MergeVFChain6Tiles
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x256x540x120xf16, {order = #NHWC}>)
func.func @MergeVFChain6Tiles(%arg0: tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_6 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_7 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_8 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_9 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_10 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_11 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_12 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_13 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs(%arg0 : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst, %cst_13) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst_12) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1, %cst_11) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst_10, %cst_9) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %cst_2, %cst_8) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %6 = VPU.NCE.Convolution(%5, %cst_3, %cst_7) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %cst_4, %cst_6) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %8 = VPU.NCE.DepthConvolution(%7, %cst_5, %cst_9) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 20]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%8 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>

    return %9: tensor<1x128x540x240xf16, {order = #NHWC}>

    //CHECK: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 44 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 960 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    //CHECK: [[CAST_INPUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {
    //CHECK:        [[OUTPUT_SIZE:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER]])
    //CHECK:        [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:        [[TEMP_VALUE1:%.+]] = affine.min #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:        [[PAD_LOW5:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:        [[TEMP_VALUE2:%.+]] = affine.min #[[$MAP4]]([[LOOP_ITER]], [[TEMP_VALUE0]])
    //CHECK:        [[PAD_HIGH5:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE2]]]
    //CHECK:        [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE0]])
    //CHECK:        [[TEMP_VALUE3:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE0]])
    //CHECK:        [[PAD_LOW4:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE3]]]
    //CHECK:        [[TEMP_VALUE4:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE0]], [[TEMP_VALUE2]])
    //CHECK:        [[PAD_HIGH4:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE4]]]
    //CHECK:        [[TEMP_VALUE5:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE2]])
    //CHECK:        [[TEMP_VALUE6:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE2]])
    //CHECK:        [[PAD_LOW3:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE6]]]
    //CHECK:        [[TEMP_VALUE7:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE2]], [[TEMP_VALUE5]])
    //CHECK:        [[PAD_HIGH3:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE7]]]
    //CHECK:        [[TEMP_VALUE8:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE5]])
    //CHECK:        [[TEMP_VALUE9:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE5]])
    //CHECK:        [[PAD_LOW2:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE9]]]
    //CHECK:        [[TEMP_VALUE10:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE5]], [[TEMP_VALUE8]])
    //CHECK:        [[PAD_HIGH2:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE10]]]
    //CHECK:        [[TEMP_VALUE11:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE8]])
    //CHECK:        [[TEMP_VALUE12:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE8]])
    //CHECK:        [[PAD_LOW1:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE12]]]
    //CHECK:        [[TEMP_VALUE13:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE8]], [[TEMP_VALUE11]])
    //CHECK:        [[PAD_HIGH1:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE13]]]
    //CHECK:        [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE11]])
    //CHECK:        [[TEMP_VALUE14:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE11]])
    //CHECK:        [[PAD_LOW0:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE14]]]
    //CHECK:        [[SLICE_SIZE:%.+]] = affine.apply #[[$MAP5]]([[OUTPUT_SIZE]])
    //CHECK:        [[TEMP_VALUE15:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE11]], [[SLICE_OFFSET]])
    //CHECK:        [[PAD_HIGH0:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE15]]]
    //CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SLICE_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {order = #NHWC}>

    //CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:        tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW3]]] high[0, 0, 1, [[PAD_HIGH3]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, 1, [[PAD_LOW4]]] high[0, 0, 1, [[PAD_HIGH4]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, 1, [[PAD_LOW5]]] high[0, 0, 1, [[PAD_HIGH5]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 540, [[OUTPUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:    scf.yield [[INSERT]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:    [[CAST:%.+]] = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs([[LOOP]]
    //CHECK:    return [[CAST]] : tensor<1x128x540x240xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 240)>

// CHECK-LABEL: @MergeDynamicEltwise
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
func.func @MergeDynamicEltwise(
         %arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
         is_inplace = true,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             quant_scale = [1.000000e+00],
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         tilingStrategy = [1, 1, 1, 2]
     } -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

     %1 = VPU.NCE.Eltwise(%0, %0) {
         is_inplace = true,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             quant_scale = [1.000000e+00],
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         tilingStrategy = [1, 1, 1, 2]
     } -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

     return %1 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 240 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_INDEX:%.+]] = arith.constant 3 : index

    //CHECK: [[DIM:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[SLICE_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[SLICE_SIZE]]] [1, 1, 1, 1]
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE]], [[SLICE]])
    //CHECK:                [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[ELTWISE0]])

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[SLICE_SIZE]]] [1, 1, 1, 1]
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[LOOP]] :  tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

IE.TileResource 3 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 960, 13)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 959)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0) -> (d0 + 1)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0) -> (-d0 + 960, 27)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0) -> (d0 + 2)>

// CHECK-LABEL: @MergeVF2Chains
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x540x960xf16, {order = #NHWC}>)
func.func @MergeVF2Chains(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_4 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_5 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_6 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_6, %cst_5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst, %cst_4) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.Sign(%1) : tensor<1x32x540x960xf16, {order = #NHWC}>  -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_0, %cst_3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.Convolution(%3, %cst_1, %cst_2) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %4: tensor<1x32x540x960xf16, {order = #NHWC}>

    //CHECK: [[LOOP_STEP1:%.+]] = arith.constant 27 : index
    //CHECK: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP0:%.+]] = arith.constant 13 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 960 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_OUTPUT0:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP0:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER0:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP0]]
    //CHECK-SAME:           iter_args([[LOOP_OUT0:%arg[0-9]]] = [[LOOP_OUTPUT0]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>)

    //CHECK:                [[INSERT_SIZE0:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER0]])
    //CHECK:                [[SLICE_OFFSET0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER0]])
    //CHECK:                [[TEMP_VALUE0:%.+]] = affine.min #[[$MAP2]]([[LOOP_ITER0]])
    //CHECK:                [[PAD_LOW0:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE0]]]
    //CHECK:                [[TEMP_VALUE1:%.+]] = affine.min #[[$MAP4]]([[LOOP_ITER0]], [[SLICE_OFFSET0]])
    //CHECK:                [[PAD_HIGH0:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:                [[SLICE_SIZE0:%.+]] = affine.apply #[[$MAP5]]([[INSERT_SIZE0]])

    //CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[SLICE_OFFSET0]]] [1, 32, 540, [[SLICE_SIZE0]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {order = #NHWC}>
    //CHECK:                [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SLICE0]]

    //CHECK:                [[PAD0:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:                [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]

    //CHECK:                [[INSERT0:%.+]] = tensor.insert_slice [[CONV0]] into [[LOOP_OUT0]][0, 0, 0, [[LOOP_ITER0]]] [1, 32, 540, [[INSERT_SIZE0]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT0]] : tensor<1x32x540x960xf16, {order = #NHWC}>

    //CHECK: [[SIGN:%.+]] = VPU.Sign([[LOOP0]]) : tensor<1x32x540x960xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT1:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP1:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER1:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP1]]
    //CHECK-SAME:           iter_args([[LOOP_OUT1:%arg[0-9]]] = [[LOOP_OUTPUT1]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>)

    //CHECK:                [[INSERT_SIZE1:%.+]] = affine.min #[[$MAP6]]([[LOOP_ITER1]])
    //CHECK:                [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER0]])
    //CHECK:                [[TEMP_VALUE3:%.+]] = affine.min #[[$MAP2]]([[LOOP_ITER0]])
    //CHECK:                [[PAD_LOW2:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE3]]]
    //CHECK:                [[TEMP_VALUE4:%.+]] = affine.min #[[$MAP4]]([[LOOP_ITER1]], [[TEMP_VALUE2]])
    //CHECK:                [[PAD_HIGH2:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE4]]]
    //CHECK:                [[SLICE_OFFSET1:%.+]]  = affine.max #[[$MAP1]]([[TEMP_VALUE2]])
    //CHECK:                [[TEMP_VALUE5:%.+]] = affine.min #[[$MAP2]]([[TEMP_VALUE2]])
    //CHECK:                [[PAD_LOW1:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE5]]]
    //CHECK:                [[SLICE_SIZE1:%.+]] = affine.apply #[[$MAP7]]([[INSERT_SIZE1]])
    //CHECK:                [[TEMP_VALUE6:%.+]] = affine.min #[[$MAP4]]([[TEMP_VALUE2]], [[SLICE_OFFSET1]])
    //CHECK:                [[PAD_HIGH1:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE6]]]

    //CHECK:                [[SLICE1:%.+]] = tensor.extract_slice [[SIGN]][0, 0, 0, [[SLICE_OFFSET1]]] [1, 32, 540, [[SLICE_SIZE1]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {order = #NHWC}>

    //CHECK:                [[PAD1:%.+]] = tensor.pad [[SLICE1]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]]
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:                [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]

    //CHECK:                [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:                [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]

    //CHECK:                [[INSERT1:%.+]] = tensor.insert_slice [[CONV2]] into [[LOOP_OUT1]][0, 0, 0, [[LOOP_ITER1]]] [1, 32, 540, [[INSERT_SIZE1]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT1]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: return [[LOOP1]] : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

 // CHECK-LABEL: @MergeWithLayoutCast
 // CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x256x140xf16>,
 // CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x256x140xf16>)
 func.func @MergeWithLayoutCast(
         %arg0: tensor<1x16x256x140xf16>,
         %arg1: tensor<1x16x256x140xf16>
 ) -> tensor<1x16x256x140xf16> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16, {order = #NHWC}>
    %1 = VPU.LayoutCast(%arg1) {dst_order = #NHWC} : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16, {order = #NHWC}>
     %2 = VPU.NCE.Eltwise(%0, %1) {
         is_inplace = true,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             quant_scale = [1.000000e+00],
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         tilingStrategy = [1, 1, 1, 2]
     } -> tensor<1x16x256x140xf16, {order = #NHWC}>

     %3 = VPU.LayoutCast(%2) {dst_order = #NCHW} : tensor<1x16x256x140xf16, {order = #NHWC}> -> tensor<1x16x256x140xf16>

     return %3 : tensor<1x16x256x140xf16>


    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 70 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 140 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x256x140xf16, {order = #NHWC}>

    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x140xf16, {order = #NHWC}>)

    //CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x140xf16> to tensor<1x16x256x70xf16>
    //CHECK:                [[CAST0:%.+]] = VPU.LayoutCast([[SLICE0]]) {dst_order = #NHWC} : tensor<1x16x256x70xf16> -> tensor<1x16x256x70xf16, {order = #NHWC}>
    //CHECK:                [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x140xf16> to tensor<1x16x256x70xf16>
    //CHECK:                [[CAST1:%.+]] = VPU.LayoutCast([[SLICE1]]) {dst_order = #NHWC} : tensor<1x16x256x70xf16> -> tensor<1x16x256x70xf16, {order = #NHWC}>
    //CHECK:                [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CAST0]], [[CAST1]])
    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x70xf16, {order = #NHWC}> into tensor<1x16x256x140xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x16x256x140xf16, {order = #NHWC}>

    // pure view-like op doesn't have tilingStrategy, it cannot be tiled. it might be used to continue VF further, but we cannot start VF with that unfortunately
    //CHECK: [[CAST2:%.+]] = VPU.LayoutCast([[LOOP]]) {dst_order = #NCHW} : tensor<1x16x256x140xf16, {order = #NHWC}> -> tensor<1x16x256x140xf16>
    //CHECK: return [[CAST2]] : tensor<1x16x256x140xf16>
}
