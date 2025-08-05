//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-tiling="enable-scf-tiling=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 - 1, 0)>

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

    //CHECK:      [[OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:      [[ZERO:%.+]] = arith.constant 0 : index
    //CHECK:      [[CONDITION:%.+]] = arith.cmpi eq, [[LOOP_ITER]], [[ZERO]] : index
    //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
    //CHECK:      [[IF:%.+]] = scf.if [[CONDITION]] -> (tensor<1x256x32x64xf16, {order = #NHWC}>) {
    //CHECK:           [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    //CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
    //CHECK:           scf.yield [[CONV0]] : tensor<1x256x32x64xf16, {order = #NHWC}>
    //CHECK:      else
    //CHECK:           [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    //CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    //CHECK:           scf.yield [[CONV1]] : tensor<1x256x32x64xf16, {order = #NHWC}>

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[IF]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 - 1, 0)>

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

    //CHECK:      [[OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:      [[ZERO:%.+]] = arith.constant 0 : index
    //CHECK:      [[CONDITION:%.+]] = arith.cmpi eq, [[LOOP_ITER]], [[ZERO]] : index
    //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 16, 101, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x101x200xf16, {order = #NHWC}>
    //CHECK:      [[IF:%.+]] = scf.if [[CONDITION]] -> (tensor<1x16x100x200xf16, {order = #NHWC}>) {
    //CHECK:           [[MAXPOOL0:%.+]] = VPU.NCE.MaxPool([[SLICE]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
    //CHECK:           scf.yield [[MAXPOOL0]] : tensor<1x16x100x200xf16, {order = #NHWC}>
    //CHECK:      else
    //CHECK:           [[MAXPOOL1:%.+]] = VPU.NCE.MaxPool([[SLICE]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    //CHECK:           scf.yield [[MAXPOOL1]] : tensor<1x16x100x200xf16, {order = #NHWC}>

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[IF]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, 100, 200] [1, 1, 1, 1] : tensor<1x16x100x200xf16, {order = #NHWC}> into tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK:   scf.yield [[INSERT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x16x200x200xf16, {order = #NHWC}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 - 1, 0)>

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

    //CHECK:       [[OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:       [[ZERO:%.+]] = arith.constant 0 : index
    //CHECK:       [[CONDITION0:%.+]] = arith.cmpi eq, [[LOOP_ITER]], [[ZERO]] : index
    //CHECK:       [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 16, 51, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x51x200xf16, {order = #NHWC}>
    //CHECK:       [[OUTER_IF:%.+]] = scf.if [[CONDITION0]] -> (tensor<1x16x50x200xf16, {order = #NHWC}>)
    //CHECK:            [[MAXPOOL0:%.+]] = VPU.NCE.MaxPool([[SLICE0]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:       pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
    //CHECK:            scf.yield [[MAXPOOL0]] : tensor<1x16x50x200xf16, {order = #NHWC}>
    //CHECK:       else
    //CHECK:            [[SIZE:%.+]] = arith.constant 200 : index
    //CHECK:            [[SUB:%.+]] = arith.subi [[SIZE]], [[LOOP_ITER]] : index
    //CHECK:            [[CONDITION1:%.+]] = arith.cmpi eq, [[LOOP_ITER]], [[SUB]] : index
    //CHECK:            [[INNER_IF:%.+]] = scf.if [[CONDITION1]] -> (tensor<1x16x50x200xf16, {order = #NHWC}>) {
    //CHECK:                 [[MAXPOOL1:%.+]] = VPU.NCE.MaxPool([[SLICE0]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    //CHECK:                 scf.yield [[MAXPOOL1]] : tensor<1x16x50x200xf16, {order = #NHWC}>
    //CHECK:            else
    //CHECK:                 [[SLICE1:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 16, 52, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x52x200xf16, {order = #NHWC}>
    //CHECK:                 [[MAXPOOL2:%.+]] = VPU.NCE.MaxPool([[SLICE1]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:                 scf.yield [[MAXPOOL2]] : tensor<1x16x50x200xf16, {order = #NHWC}>
    //CHECK:            scf.yield [[INNER_IF]] : tensor<1x16x50x200xf16, {order = #NHWC}>

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[OUTER_IF]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, 50, 200] [1, 1, 1, 1] : tensor<1x16x50x200xf16, {order = #NHWC}> into tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
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
