//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-tiling="enable-scf-tiling=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 - 1, 0)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (-(d0 - 1), 0)>
//CHECK: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0, 1)>
//CHECK: #[[$MAP3:.*]] = affine_map<(d0, d1) -> (d1 + d0 + 1 - 64, 0)>

// CHECK-LABEL:   @ApplyTilingNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @ApplyTilingNCEConv(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32, {order = #NCHW}> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 64 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 32 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {

    //CHECK:                [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:                [[DIFF1:%.+]] = affine.min #[[$MAP1]](%arg1)
    //CHECK:                [[PAD_LOW:%.+]] = affine.max #[[$MAP2]]()[[[DIFF1]]]
    //CHECK:                [[DIFF2:%.+]] = affine.min #[[$MAP3]](%arg1, [[SLICE_OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.max #[[$MAP2]]()[[[DIFF2]]]

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
    //CHECK:                [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x66xf16, {order = #NHWC}>
    //CHECK:                [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    //CHECK-SAME:           {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           tensor<1x32x?x66xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32, {order = #NCHW}> -> tensor<1x256x?x64xf16, {order = #NHWC}>
    //CHECK:                [[RESULT_SIZE:%.+]] = arith.constant 32 : index

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 256, [[RESULT_SIZE]], 64] [1, 1, 1, 1] : tensor<1x256x?x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 - 1, 0)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (-(d0 - 1), 0)>
//CHECK: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0, 1)>
//CHECK: #[[$MAP3:.*]] = affine_map<(d0, d1) -> (d1 + d0 + 1 - 200, 0)>

// CHECK-LABEL: @ApplyTilingMaxPool
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func.func @ApplyTilingMaxPool(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 200 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 100 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x200x200xf16, {order = #NHWC}>) {

    //CHECK:                [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:                [[DIFF1:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.max #[[$MAP2]]()[[[DIFF1]]]
    //CHECK:                [[DIFF2:%.+]] = affine.min #[[$MAP3]]([[LOOP_ITER]], [[SLICE_OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.max #[[$MAP2]]()[[[DIFF2]]]
    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 16, 101, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x101x200xf16, {order = #NHWC}>
    //CHECK:                [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x16x101x200xf16, {order = #NHWC}> to tensor<1x16x?x202xf16, {order = #NHWC}>
    //CHECK:                [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[PAD]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           tensor<1x16x?x200xf16, {order = #NHWC}>
    //CHECK:                [[RESULT_SIZE:%.+]] = arith.constant 100 : index

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[MAXPOOL]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, [[RESULT_SIZE]], 200] [1, 1, 1, 1] : tensor<1x16x?x200xf16, {order = #NHWC}> into tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x16x200x200xf16, {order = #NHWC}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 - 1, 0)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-(d0 - 1), 0)>
//CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0, 1)>
//CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d1 + d0 + 1 - 200, 0)>

// CHECK-LABEL: @ApplyTilingMaxPool4Tiles
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func.func @ApplyTilingMaxPool4Tiles(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 4, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 200 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 50 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x200x200xf16, {order = #NHWC}>) {

    //CHECK:                [[OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:                [[DIFF1:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.max #[[$MAP2]]()[[[DIFF1]]]
    //CHECK:                [[DIFF2:%.+]] = affine.min #[[$MAP3]]([[LOOP_ITER]], [[OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.max #[[$MAP2]]()[[[DIFF2]]]

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 16, 51, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x51x200xf16, {order = #NHWC}>
    //CHECK:                [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x16x51x200xf16, {order = #NHWC}> to tensor<1x16x?x202xf16, {order = #NHWC}>

    //CHECK:                [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[PAD]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           tensor<1x16x?x200xf16, {order = #NHWC}>
    //CHECK:                [[RESULT_SIZE:%.+]] = arith.constant 50 : index

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[MAXPOOL]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, [[RESULT_SIZE]], 200] [1, 1, 1, 1] : tensor<1x16x?x200xf16, {order = #NHWC}> into tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x16x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingAvgPool
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x12960xf16, {order = #NHWC}>
func.func @ApplyTilingAvgPool(%arg0: tensor<1x16x7x12960xf16, {order = #NHWC}>) -> tensor<1x16x1x12960xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [7, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 3]
        } -> tensor<1x16x1x12960xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x12960xf16, {order = #NHWC}>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x1x12960xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 12960 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 4320 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x1x12960xf16, {order = #NHWC}>) {

    //CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 7, 4320] [1, 1, 1, 1] : tensor<1x16x7x12960xf16, {order = #NHWC}> to tensor<1x16x7x4320xf16, {order = #NHWC}>
    //CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[SLICE]])

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[AVGPOOL]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 1, 4320] [1, 1, 1, 1] : tensor<1x16x1x4320xf16, {order = #NHWC}> into tensor<1x16x1x12960xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x16x1x12960xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x16x1x12960xf16, {order = #NHWC}>

}


// -----


 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

 // CHECK-LABEL: @NoPaddingDWCONV
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x32x200x200xf16, {order = #NHWC}>,
 // CHECK-SAME:      [[WEIGHTS:%arg[0-9]]]: tensor<32x16x1x1xf16, {order = #NHWC}>,
 // CHECK-SAME:      [[WEIGHTS_TABLE:%arg[0-9]]]: tensor<32x1x1x4xsi32>
 func.func @NoPaddingDWCONV(
         %arg0: tensor<1x32x200x200xf16, {order = #NHWC}>,
         %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>,
         %arg2: tensor<32x1x1x4xsi32>
 ) -> tensor<1x32x200x200xf16, {order = #NHWC}> {
     %1 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) {
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64
         >,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         rawFilterShape = [32, 1, 1, 1],
         strides = [1, 1],
         tilingStrategy = [1, 1, 1, 4]
     } -> tensor<1x32x200x200xf16, {order = #NHWC}>

     return %1 : tensor<1x32x200x200xf16, {order = #NHWC}>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x200x200xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 200 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 50 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x32x200x200xf16, {order = #NHWC}>) {

    //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 200, 50] [1, 1, 1, 1] : tensor<1x32x200x200xf16, {order = #NHWC}> to tensor<1x32x200x50xf16, {order = #NHWC}>
    //CHECK:      [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[SLICE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[DWCONV]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 200, 50] [1, 1, 1, 1] : tensor<1x32x200x50xf16, {order = #NHWC}> into tensor<1x32x200x200xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x32x200x200xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x32x200x200xf16, {order = #NHWC}>
}

// -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

 // CHECK-LABEL: @NotPaddedMaxPool
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x256x480xf16, {order = #NHWC}>
 func.func @NotPaddedMaxPool(
         %arg0: tensor<1x16x256x480xf16, {order = #NHWC}>
 ) -> tensor<1x16x127x480xf16, {order = #NHWC}> {
     %1 = VPU.NCE.MaxPool(%arg0) {
         kernel_size = [3, 1],
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64>,
             ppe = #VPU.PPEInt<mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [2, 1],
         tilingStrategy = [1, 1, 1, 4]
     } -> tensor<1x16x127x480xf16, {order = #NHWC}>

     return %1 : tensor<1x16x127x480xf16, {order = #NHWC}>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 480 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 120 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x127x480xf16, {order = #NHWC}>) {

    //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 120] [1, 1, 1, 1] : tensor<1x16x256x480xf16, {order = #NHWC}> to tensor<1x16x256x120xf16, {order = #NHWC}>
    //CHECK:      [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SLICE]])
    //CHECK-SAME: pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    //CHECK: [[INSERT:%.+]]  = tensor.insert_slice [[MAXPOOL]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 127, 120] [1, 1, 1, 1] : tensor<1x16x127x120xf16, {order = #NHWC}> into tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK:  return [[LOOP]] : tensor<1x16x127x480xf16, {order = #NHWC}>
 }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

 // CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (47, -d0 + 140)>
 // CHECK-LABEL: @UnEvenEltwiseTiling
 // CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x256x140xf16, {order = #NHWC}>,
 // CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x256x140xf16, {order = #NHWC}>)
 func.func @UnEvenEltwiseTiling(
         %arg0: tensor<1x16x256x140xf16, {order = #NHWC}>,
         %arg1: tensor<1x16x256x140xf16, {order = #NHWC}>
 ) -> tensor<1x16x256x140xf16, {order = #NHWC}> {
     %1 = VPU.NCE.Eltwise(%arg0, %arg1) {
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
         tilingStrategy = [1, 1, 1, 3]
     } -> tensor<1x16x256x140xf16, {order = #NHWC}>

     return %1 : tensor<1x16x256x140xf16, {order = #NHWC}>


    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x256x140xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 140 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 47 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x140xf16, {order = #NHWC}>) {

    //CHECK:      [[UNEVEN_SIZE:%.+]]  = affine.min #[[$MAP]]([[LOOP_ITER]])
    //CHECK:      [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] : tensor<1x16x256x140xf16, {order = #NHWC}> to tensor<1x16x256x?xf16, {order = #NHWC}>
    //CHECK:      [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] : tensor<1x16x256x140xf16, {order = #NHWC}> to tensor<1x16x256x?xf16, {order = #NHWC}>
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]])

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {order = #NHWC}> into tensor<1x16x256x140xf16, {order = #NHWC}>
    //CHECK:   scf.yield [[INSERT]] : tensor<1x16x256x140xf16, {order = #NHWC}>

    //CHECK: return [[LOOP]] : tensor<1x16x256x140xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
 // CHECK-LABEL: @ApplyConvCTiling
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
func.func @ApplyConvCTiling(
            %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
                -> tensor<1x512x14x14xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<1> : tensor<512x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 256, 3, 3],
            strides = [1, 1],
            tilingStrategy = [1, 2, 1, 1]
        } : tensor<1x256x14x14xf16, {order = #NHWC}>, tensor<512x256x3x3xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32, {order = #NCHW}> -> tensor<1x512x14x14xf16, {order = #NHWC}>

        return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<1> : tensor<512x1x1x4xsi32>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x512x14x14xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 512 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 256 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x512x14x14xf16, {order = #NHWC}>) {

    //CHECK:      [[SLICE_WEIGHTS:%.+]]  = tensor.extract_slice [[WEIGHTS]][[[LOOP_ITER]], 0, 0, 0] [256, 256, 3, 3] [1, 1, 1, 1] : tensor<512x256x3x3xf16, {order = #NHWC}> to tensor<256x256x3x3xf16, {order = #NHWC}>
    //CHECK:      [[SLICE_WEIGHTS_TABLE:%.+]] = tensor.extract_slice [[WEIGHTS_TABLE]][[[LOOP_ITER]], 0, 0, 0] [256, 1, 1, 4] [1, 1, 1, 1] : tensor<512x1x1x4xsi32, {order = #NCHW}> to tensor<256x1x1x4xsi32>
    //CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICE_WEIGHTS]], [[SLICE_WEIGHTS_TABLE]])

    //CHECK:      [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, 256, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf16, {order = #NHWC}> into tensor<1x512x14x14xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x512x14x14xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$MAP_MIN:.+]] = affine_map<(d0)[s0] -> (240, -d0 + s0)>

// CHECK-LABEL: @DynamicEltwiseTiling
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicEltwiseTiling(
         %arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
         %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
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

     return %0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[DIM_VALUE0:%.+]] = arith.constant 3 : index
    //CHECK: [[DIM0:%.+]] = tensor.dim [[INPUT0]], [[DIM_VALUE0]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM0]]) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_VALUE1:%.+]]  = arith.constant 3 : index
    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT0]], [[DIM_VALUE1]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 240 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:      [[UNEVEN_SIZE:%.+]] = affine.min #[[$MAP_MIN]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:      [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] :
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:           to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] :
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:           to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]])
    //CHECK:      [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] :
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:           into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      scf.yield [[INSERT]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: return [[LOOP]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
 // CHECK: #[[$MAP_MIN:.+]] = affine_map<(d0) -> (69, -d0 + 480)>

 // CHECK-LABEL: @NotPaddedUnevenMaxPool
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x256x480xf16, {order = #NHWC}>
 func.func @NotPaddedUnevenMaxPool(
         %arg0: tensor<1x16x256x480xf16, {order = #NHWC}>
 ) -> tensor<1x16x127x480xf16, {order = #NHWC}> {
     %1 = VPU.NCE.MaxPool(%arg0) {
         kernel_size = [3, 1],
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64>,
             ppe = #VPU.PPEInt<mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [2, 1],
         tilingStrategy = [1, 1, 1, 7]
     } -> tensor<1x16x127x480xf16, {order = #NHWC}>

     return %1 : tensor<1x16x127x480xf16, {order = #NHWC}>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 480 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 69 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x127x480xf16, {order = #NHWC}>) {

    //CHECK:      [[UNEVEN_SIZE:%.+]] = affine.min #[[$MAP_MIN]]([[LOOP_ITER]])
    //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] : tensor<1x16x256x480xf16, {order = #NHWC}> to tensor<1x16x256x?xf16, {order = #NHWC}>
    //CHECK:      [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SLICE]])
    //CHECK-SAME: pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    //CHECK: [[INSERT:%.+]]  = tensor.insert_slice [[MAXPOOL]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 127, [[UNEVEN_SIZE]]] [1, 1, 1, 1] : tensor<1x16x127x?xf16, {order = #NHWC}> into tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK:  return [[LOOP]] : tensor<1x16x127x480xf16, {order = #NHWC}>
 }

// -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

 func.func @NotTileLayoutCast(
         %arg0: tensor<1x16x127x480xf16>
 ) -> tensor<1x16x127x480xf16, {order = #NHWC}> {
     %0 = VPU.LayoutCast(%arg0) {dst_order = #NHWC, tilingStrategy = [1, 1, 1, 2]} : tensor<1x16x127x480xf16> -> tensor<1x16x127x480xf16, {order = #NHWC}>

     return %0 : tensor<1x16x127x480xf16, {order = #NHWC}>

    //CHECK-NOT: scf.for
 }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 - 1, 0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-(d0 - 1), 0)>
// CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0, 1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d1 + d0 + 1 - 512, 0)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (d1 + d0 + 1 - 480, 0)>

// CHECK-LABEL:   @Tiling2DNotPaddedMaxPool
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x512x480xf16, {order = #NHWC}>
 func.func @Tiling2DNotPaddedMaxPool(
         %arg0: tensor<1x16x512x480xf16, {order = #NHWC}>
 ) -> tensor<1x16x512x480xf16, {order = #NHWC}> {
     %1 = VPU.NCE.MaxPool(%arg0) {
         kernel_size = [3, 3],
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<
             left = 1 : i64,
             right = 1 : i64,
             top = 1 : i64,
             bottom = 1 : i64>,
             ppe = #VPU.PPEInt<mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [1, 1],
         tilingStrategy = [1, 1, 2, 4]
     } -> tensor<1x16x512x480xf16, {order = #NHWC}>

     return %1 : tensor<1x16x512x480xf16, {order = #NHWC}>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x512x480xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END_H:%.+]]  = arith.constant 512 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 256 : index

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x512x480xf16, {order = #NHWC}>)

    //CHECK:                [[LOOP_END_W:%.+]] = arith.constant 480 : index
    //CHECK:                [[LOOP_STEP_W:%.+]] = arith.constant 120 : index
    //CHECK:                [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:                            [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:                            iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x16x512x480xf16, {order = #NHWC}>)

    //CHECK:                                 [[SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_H]])
    //CHECK:                                 [[TEMP_VALUE0:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_H]])
    //CHECK:                                 [[PAD_LOW_H:%.+]] = affine.max #[[$MAP2]]()[[[TEMP_VALUE0]]]
    //CHECK:                                 [[TEMP_VALUE1:%.+]] = affine.min #[[$MAP3]]([[LOOP_ITER_H]], [[SLICE_OFFSET_H]])
    //CHECK:                                 [[PAD_LOW_W:%.+]] = affine.max #[[$MAP2]]()[[[TEMP_VALUE1]]]
    //CHECK:                                 [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_W]])
    //CHECK:                                 [[TEMP_VALUE2:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])
    //CHECK:                                 [[PAD_HIGH_H:%.+]] = affine.max #[[$MAP2]]()[[[TEMP_VALUE2]]]
    //CHECK:                                 [[TEMP_VALUE3:%.+]] = affine.min #[[$MAP4]]([[LOOP_ITER_W]], [[SLICE_OFFSET_W]])
    //CHECK:                                 [[PAD_HIGH_W:%.+]] = affine.max #[[$MAP2]]()[[[TEMP_VALUE3]]]

    //CHECK:                                 [[SLICE:%.+]]  = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 16, 257, 121] [1, 1, 1, 1] : tensor<1x16x512x480xf16, {order = #NHWC}> to tensor<1x16x257x121xf16, {order = #NHWC}>
    //CHECK:                                 [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK:                                 [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW_H]], [[PAD_HIGH_H]]] high[0, 0, [[PAD_LOW_W]], [[PAD_HIGH_W]]] {
    //CHECK:                                 tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                                 tensor<1x16x257x121xf16, {order = #NHWC}> to tensor<1x16x?x?xf16, {order = #NHWC}>
    //CHECK:                                 [[POOL:%.+]] = VPU.NCE.MaxPool([[PAD]])
    //CHECK:                                 [[PAD_DIM_H:%.+]] = arith.constant 256 : index
    //CHECK:                                 [[PAD_DIM_W:%.+]] = arith.constant 120 : index
    //CHECK:                                 [[INSERT:%.+]] = tensor.insert_slice [[POOL]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[PAD_DIM_H]], [[PAD_DIM_W]]] [1, 1, 1, 1]
    //CHECK-SAME:                            tensor<1x16x?x?xf16, {order = #NHWC}> into tensor<1x16x512x480xf16, {order = #NHWC}>

    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x512x480xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[LOOP_W]] : tensor<1x16x512x480xf16, {order = #NHWC}>
    //CHECK:  return [[LOOP_H]] : tensor<1x16x512x480xf16, {order = #NHWC}>
 }

 // -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 - 1, 0)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-(d0 - 1), 0)>
//CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0, 1)>
//CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d1 + d0 + 1 - 64, 0)>

// CHECK-LABEL:   @ConvChannel2DTiling
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @ConvChannel2DTiling(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 512, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 2, 8, 1]
    } : tensor<1x512x64x64xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END_H:%.+]] = arith.constant 64 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 8 : index

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>)

    //CHECK:                [[LOOP_END_C:%.+]] = arith.constant 256 : index
    //CHECK:                [[LOOP_STEP_C:%.+]] = arith.constant 128 : index
    //CHECK:                [[LOOP_C:%.+]] = scf.for
    //CHECK-SAME:                            [[LOOP_ITER_C:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_C]] step [[LOOP_STEP_C]]
    //CHECK-SAME:                            iter_args([[LOOP_OUT_C:%arg[0-9]]]  = [[LOOP_OUT]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>)

    //CHECK:                                  [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_H]])
    //CHECK:                                  [[TEMP_VALUE0:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_H]])
    //CHECK:                                  [[PAD_LOW:%.+]] = affine.max #[[$MAP2]]()[[[TEMP_VALUE0]]]
    //CHECK:                                  [[TEMP_VALUE1:%.+]] = affine.min #[[$MAP3]]([[LOOP_ITER_H]], [[SLICE_OFFSET]])
    //CHECK:                                  [[PAD_HIGH:%.+]] = affine.max #[[$MAP2]]()[[[TEMP_VALUE1]]]

    //CHECK:                                  [[SLICE_INPUT:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 512, 9, 64] [1, 1, 1, 1] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
    //CHECK:                                  [[SLICE_WEIGHTS:%.+]] = tensor.extract_slice [[WEIGHTS]][[[LOOP_ITER_C]], 0, 0, 0] [128, 512, 3, 3] [1, 1, 1, 1] : tensor<256x512x3x3xf16, {order = #NHWC}> to tensor<128x512x3x3xf16, {order = #NHWC}>
    //CHECK:                                  [[SLICE_WEIGHTS_TABLE:%.+]] = tensor.extract_slice [[WEIGHTS_TABLE]][[[LOOP_ITER_C]], 0, 0, 0] [128, 1, 1, 4] [1, 1, 1, 1] : tensor<256x1x1x4xsi32> to tensor<128x1x1x4xsi32>
    //CHECK:                                  [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK:                                  [[PAD:%.+]] = tensor.pad [[SLICE_INPUT]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                                  tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                                  tensor<1x512x9x64xf16, {order = #NHWC}> to tensor<1x512x?x66xf16, {order = #NHWC}>

    //CHECK:                                  [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[SLICE_WEIGHTS]], [[SLICE_WEIGHTS_TABLE]])
    //CHECK:                                  [[PADDED_DIM:%.+]] = arith.constant 8 : index
    //CHECK:                                  [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT_C]][0, [[LOOP_ITER_C]], [[LOOP_ITER_H]], 0] [1, 128, [[PADDED_DIM]], 64] [1, 1, 1, 1] : tensor<1x128x?x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[LOOP_C]]
    //CHECK:  return [[LOOP_H]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (128, -d0 + s0)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (240, -d0 + s0)>

// CHECK-LABEL: @Dynamic2DEltwiseTiling
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
func.func @Dynamic2DEltwiseTiling(
         %arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
         %arg1: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
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
         tilingStrategy = [1, 1, 2, 2]
     } -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

     return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[DIM_VALUE_H_0:%.+]] = arith.constant 2 : index
    //CHECK: [[DIM_H_0:%.+]] = tensor.dim [[INPUT0]], [[DIM_VALUE_H_0]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[DIM_VALUE_W_0:%.+]] = arith.constant 3 : index
    //CHECK: [[DIM_W_0:%.+]] = tensor.dim [[INPUT0]], [[DIM_VALUE_W_0]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H_0]], [[DIM_W_0]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_VALUE_H_1:%.+]] = arith.constant 2 : index
    //CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT0]], [[DIM_VALUE_H_1]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[DIM_VALUE_W_1:%.+]] = arith.constant 3 : index
    //CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT0]], [[DIM_VALUE_W_1]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 128 : index
    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
    //CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 240 : index

    //CHECK:                  [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:                              [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:                              iter_args([[LOOP_OUT_W:%arg[0-9]]]  = [[LOOP_OUT]])

    //CHECK:                                   [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    //CHECK:                                   [[SLICE_SIZE_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]
    //CHECK:                                   [[SLICE_INPUT0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK:                                   [[SLICE_INPUT1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK:                                   [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE_INPUT0]], [[SLICE_INPUT1]])
    //CHECK:                                   [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:                              tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[LOOP_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[LOOP_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (512, -d0 + s0)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0 - 1, 0)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-(d0 - 1), 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (s0, 1)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0) -> (d0 + 1)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1) -> (d1 + d0 + 1 - -9223372036854775808, 0)>

// CHECK-LABEL:   @ApplyTilingNCEConvDyn
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingNCEConvDyn(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32, {order = #NCHW}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>
    //CHECK: [[C2:%.+]] = arith.constant 2 : index
    //CHECK: [[DIM:%.+]] = tensor.dim [[INPUT]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[C0:%.+]] = arith.constant 0 : index
    //CHECK: [[C1:%.+]] = arith.constant 1 : index
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM]]) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_INDEX:%.+]] = arith.constant 2 : index
    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[C0_1:%.+]] = arith.constant 0 : index
    //CHECK: [[C1_1:%.+]] = arith.constant 1 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 512 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[RESULT_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[TEMP_VALUE0:%.+]] = affine.min #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE0]]]
    //CHECK:                [[STRIDE_OFFSET:%.+]] = affine.apply #[[$MAP4]]([[RESULT_SIZE]])
    //CHECK:                [[TEMP_VALUE1:%.+]] = affine.min #[[$MAP5]]([[LOOP_ITER]], [[SLICE_OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE1]]]

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, [[STRIDE_OFFSET]], 64] [1, 1, 1, 1]
    //CHECK-SAME:           : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 66]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    //CHECK-SAME:           {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 66]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32, {order = #NCHW}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 64]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 256, [[RESULT_SIZE]], 64] [1, 1, 1, 1] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (100, -d0 + s0)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 - 1, 0)>
//CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (-(d0 - 1), 0)>
//CHECK: #[[$MAP3:.+]] = affine_map<()[s0] -> (s0, 1)>
//CHECK: #[[$MAP4:.+]] = affine_map<(d0) -> (d0 + 1)>
//CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (d1 + d0 + 1 - -9223372036854775808, 0)>
// CHECK-LABEL: @ApplyTilingMaxPool4Tiles
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>)
func.func @ApplyTilingMaxPool4Tiles(%arg0: tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 4, 1]
    } -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK: [[C2:%.+]] = arith.constant 2 : index
    //CHECK: [[DIM:%.+]] = tensor.dim [[INPUT]], [[C2]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[C0:%.+]] = arith.constant 0 : index
    //CHECK: [[C1:%.+]] = arith.constant 1 : index
    //CHECK: [[OUTPUT:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_START:%.+]] = arith.constant 0 : index
    //CHECK: [[INDEX:%.+]] = arith.constant 2 : index
    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[INDEX]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[C0_3:%.+]] = arith.constant 0 : index
    //CHECK: [[C1_4:%.+]] = arith.constant 1 : index
    //CHECK: [[STEP:%.+]] = arith.constant 100 : index
    //CHECK: [[RESULT:%.+]] = scf.for [[LOOP_ITER:%.+]] = [[LOOP_START]] to [[LOOP_END]] step [[STEP]] iter_args([[LOOP_OUT:%.+]] = [[OUTPUT]]) -> (tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>) {
    //CHECK:                [[MIN_OFFSET:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[OFFSET:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[TEMP_VALUE0:%.+]] = affine.min #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE0]]]
    //CHECK:                [[SIZE:%.+]] = affine.apply #[[$MAP4]]([[MIN_OFFSET]])
    //CHECK:                [[TEMP_VALUE1:%.+]] = affine.min #[[$MAP5]]([[LOOP_ITER]], [[OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.max #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 16, [[SIZE]], 200] [1, 1, 1, 1] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE0]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                    ^bb0([[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index, [[ARG6:%.+]]: index):
    //CHECK:                    tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                } : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x202xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 202]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[POOL_RESULT:%.+]] = VPU.NCE.MaxPool([[PAD]], [[WEIGHTS_TABLE]] ) {kernel_size = [3, 3], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[SLICE1:%.+]] = tensor.insert_slice [[POOL_RESULT]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, [[MIN_OFFSET]], 200] [1, 1, 1, 1] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[SLICE1]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: }
    //CHECK: return [[RESULT]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
}
