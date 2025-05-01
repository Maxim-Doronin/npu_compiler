//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0043085547638874429:24>

// CHECK-LABEL: @DontTileD2SDMA
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x64x128x128x!qElemType, {order = #NHWC}>
func.func @DontTileD2SDMA(%arg0: tensor<1x64x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]}
            -> tensor<1x64x128x128x!qElemType, {order = #NHWC}>
    %d2s = VPU.DepthToSpace(%avgpool) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x128x128x!qElemType, {order = #NHWC}>
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    %eltwise = VPU.NCE.Eltwise(%d2s, %d2s) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>,
        tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    return %eltwise : tensor<1x16x256x256x!qElemType, {order = #NHWC}>


    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[AVGPOOL]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 1]
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[D2S]], [[D2S]])
    // CHECK:       return [[ELTWISE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0043085547638874429:24>

// CHECK-LABEL: @DontTileD2SDMAWithSlice
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x64x128x129x!qElemType, {order = #NHWC}>
func.func @DontTileD2SDMAWithSlice(%arg0: tensor<1x64x128x129x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]}
            -> tensor<1x64x128x129x!qElemType, {order = #NHWC}>
    %slice = VPU.Slice %avgpool [0, 0, 0, 0] [1, 64, 128, 128] : tensor<1x64x128x129x!qElemType, {order = #NHWC}> to tensor<1x64x128x128x!qElemType, {order = #NHWC}>
    %d2s = VPU.DepthToSpace(%slice) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x128x128x!qElemType, {order = #NHWC}>
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    %eltwise = VPU.NCE.Eltwise(%d2s, %d2s) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>,
        tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    return %eltwise : tensor<1x16x256x256x!qElemType, {order = #NHWC}>


    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
    // CHECK:       [[SLICE:%.+]] = VPU.Slice [[AVGPOOL]]
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[SLICE]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 1]
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[D2S]], [[D2S]])
    // CHECK:       return [[ELTWISE]]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndGTile
func.func @NCEMatMulSOGAndGTile(%arg0: tensor<64x8x64x32xf16>, %arg1: tensor<64x8x64x32xf16>) -> tensor<512x1x64x64x1xf16, {order = #GNHWC}> {
  %cst_0 = const.Declare tensor<512x64x1x1x4xsi32> = dense<10> : tensor<512x64x1x1x4xsi32>
  %0 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg0 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %1 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg1 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]} : tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst_0) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 64, 32, 1, 1], strides = [1, 1], tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]} : tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
  return %8 : tensor<512x1x64x64x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [2, 1, 1, 1, 1]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndHTile
func.func @NCEMatMulSOGAndHTile(%arg0: tensor<6x1x512x512xf16>, %arg1: tensor<6x1x512x512xf16>) -> tensor<6x1x512x512x1xf16, {order = #GNHWC}> {
  %cst = const.Declare tensor<6x512x1x1x4xsi32> = dense<10> : tensor<6x512x1x1x4xsi32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<6x512x512x1x1xf16> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<6x512x512x1x1xf16> -> tensor<6x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [6, 1, 512, 128, 4]} : tensor<6x1x512x512x1xf16, {order = #GNHWC}> -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [6, 512, 512, 1, 1], strides = [1, 1]} -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [6, 1, 512, 512, 1]} : tensor<6x1x512x128x4xf16, {order = #GNHWC}> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<6x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [1, 1, 1, 3, 1]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndGHTile
func.func @NCEMatMulSOGAndGHTile(%arg0: tensor<12x1x512x512xf16>, %arg1: tensor<12x1x512x512xf16>) -> tensor<12x1x512x512x1xf16, {order = #GNHWC}> {
  %cst = const.Declare tensor<12x512x1x1x4xsi32> = dense<10> : tensor<12x512x1x1x4xsi32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 12, 512, 512]} : tensor<12x1x512x512xf16> -> tensor<1x12x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 12, 512, 512]} : tensor<12x1x512x512xf16> -> tensor<1x12x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [12, 512, 512, 1, 1]} : tensor<1x12x512x512xf16> -> tensor<12x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<12x512x512x1x1xf16> -> tensor<12x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [12, 512, 512, 1, 1]} : tensor<1x12x512x512xf16> -> tensor<12x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<12x512x512x1x1xf16> -> tensor<12x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [12, 1, 512, 128, 4]} : tensor<12x1x512x512x1xf16, {order = #GNHWC}> -> tensor<12x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [12, 512, 512, 1, 1], strides = [1, 1]} -> tensor<12x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [12, 1, 512, 512, 1]} : tensor<12x1x512x128x4xf16, {order = #GNHWC}> -> tensor<12x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<12x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [2, 1, 1, 2, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
  IE.TileResource 4 of @NCE at 1.850000e+03 MHz {
        IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  }

  // CHECK-LABEL: @pipeliningTilingForBigFilter
  func.func @pipeliningTilingForBigFilter(%arg0: tensor<1x1536x1x1xf16, {order = #NHWC}>, %arg1: tensor<8160x1536x1x1x!quant.uniform<i8:f16, 1.000000e+00>, {order = #NHWC}>) -> tensor<1x8160x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<8160x1x1x4xsi32> = dense<10> : tensor<8160x1x1x4xsi32>
    %310 = VPU.NCE.Convolution(%arg0, %arg1, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [8160, 1536, 1, 1], strides = [1, 1]} : tensor<1x1536x1x1xf16, {order = #NHWC}>, tensor<8160x1536x1x1x!quant.uniform<i8:f16, 1.000000e+00>, {order = #NHWC}>, tensor<8160x1x1x4xsi32> -> tensor<1x8160x1x1xf16, {order = #NHWC}>
    return %310 : tensor<1x8160x1x1xf16, {order = #NHWC}>

    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:    tilingStrategy = [1, 6, 1, 1]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @DepthToSpacePrefetchNotPossible
func.func @DepthToSpacePrefetchNotPossible(%arg0: tensor<1x8x27x40xf16>) -> tensor<1x256x54x80xf16> {
  %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<0> : tensor<1024x1x1x4xsi32>
  %cst_0 = const.Declare tensor<1024x16x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<1024x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
  %cst_1 = const.Declare tensor<1024x1x1x256xi1> = dense<0.0> : tensor<1024x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  %0 = VPU.GroupSparseTensor(%cst_0, %cst_1) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<1024xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<1024x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<1024x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<1024xi64>, alignment = 16 : i64>>
  %1 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<1x8x27x40xf16> -> tensor<1x8x27x48xf16>
  %2 = VPU.NCE.Permute(%1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x27x48xf16, {order = #NHWC}>
  %3 = VPU.Slice %2 [0, 0, 0, 0] [1, 16, 27, 40] : tensor<1x16x27x48xf16, {order = #NHWC}> to tensor<1x16x27x40xf16, {order = #NHWC}>
  %4 = VPU.NCE.Convolution(%3, %0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [1024, 16, 3, 3], strides = [1, 1]} : tensor<1x16x27x40xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<1024x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<1024x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<1024xi64>, alignment = 16 : i64>>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x27x40xf16, {order = #NHWC}>
  %5 = VPU.DepthToSpace(%4) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x27x40xf16, {order = #NHWC}> -> tensor<1x256x54x80xf16, {order = #NHWC}>
  %6 = VPU.NCE.MaxPool(%5) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x256x54x80xf16>
  return %6 : tensor<1x256x54x80xf16>

  // CHECK:       VPU.DepthToSpace
  // CHECK-SAME:  tilingStrategy = [1, 1, 1, 1]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:  @DepthToSpacefillDividedTilesCorrectly
func.func @DepthToSpacefillDividedTilesCorrectly(%arg0: tensor<1x8x23x40xf16>) -> tensor<1x200x46x80xf16> {
  %cst = const.Declare tensor<3200x1x1x4xsi32> = dense<0> : tensor<3200x1x1x4xsi32>
  %cst_0 = const.Declare tensor<3200x32x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<3200x32x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
  %cst_1 = const.Declare tensor<3200x1x1x384xi1> = dense<0.0> : tensor<3200x32x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  %0 = VPU.GroupSparseTensor(%cst_0, %cst_1) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<3200xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<3200x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<3200x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<3200xi64>, alignment = 16 : i64>>
  %1 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<1x8x23x40xf16> -> tensor<1x8x23x48xf16>
  %2 = VPU.NCE.Permute(%1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 8 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x8x23x48xf16, {order = #NHWC}>
  %3 = VPU.Slice %2 [0, 0, 0, 0] [1, 8, 23, 40] : tensor<1x8x23x48xf16, {order = #NHWC}> to tensor<1x8x23x40xf16, {order = #NHWC}>
  %4 = VPU.ShapeCast {shape = [1, 32, 23, 10]} inputs(%3 : tensor<1x8x23x40xf16, {order = #NHWC}>) -> tensor<1x32x23x10xf16, {order = #NHWC}>
  %5 = VPU.NCE.Convolution(%4, %0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [3200, 32, 3, 3], strides = [1, 1]} : tensor<1x32x23x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<3200x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<3200x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<3200xi64>, alignment = 16 : i64>>, tensor<3200x1x1x4xsi32> -> tensor<1x3200x23x10xf16, {order = #NHWC}>
  %6 = VPU.ShapeCast {shape = [1, 800, 23, 40]} inputs(%5 : tensor<1x3200x23x10xf16, {order = #NHWC}>) -> tensor<1x800x23x40xf16, {order = #NHWC}>
  %7 = VPU.DepthToSpace(%6) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x800x23x40xf16, {order = #NHWC}> -> tensor<1x200x46x80xf16, {order = #NHWC}>
  %8 = VPU.ShapeCast {shape = [1, 16, 46, 1000]} inputs(%7 : tensor<1x200x46x80xf16, {order = #NHWC}>) -> tensor<1x16x46x1000xf16, {order = #NHWC}>
  %9 = VPU.NCE.MaxPool(%8) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x46x1000xf16, {order = #NWCH}>
  %10 = VPU.LayoutCast(%9) {dst_order = #NHWC} : tensor<1x16x46x1000xf16, {order = #NWCH}> -> tensor<1x16x46x1000xf16, {order = #NHWC}>
  %11 = VPU.ShapeCast {shape = [1, 16, 80, 575]} inputs(%10 : tensor<1x16x46x1000xf16, {order = #NHWC}>) -> tensor<1x16x80x575xf16, {order = #NHWC}>
  %12 = VPU.NCE.MaxPool(%11) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x80x575xf16, {order = #NWCH}>
  %13 = VPU.PermuteCast(%12) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x80x575xf16, {order = #NWCH}> -> tensor<1x575x16x80xf16>
  %14 = VPU.ShapeCast {shape = [1, 200, 46, 80]} inputs(%13 : tensor<1x575x16x80xf16>) -> tensor<1x200x46x80xf16>
  return %14 : tensor<1x200x46x80xf16>

  // CHECK:       VPU.DepthToSpace
  // CHECK-SAME:  tilingStrategy = [1, 1, 3, 1]
}
