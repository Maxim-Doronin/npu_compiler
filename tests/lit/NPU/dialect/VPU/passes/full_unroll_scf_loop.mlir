//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --full-unroll-scf-loop %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (-d0 + 2560, 38)>

// CHECK: func.func @UnrollConvert([[INPUT:%.+]]: tensor<1x4x1600x2560xf32, {order = #NHWC}>)
func.func @UnrollConvert(%arg0: tensor<1x4x1600x2560xf32, {order = #NHWC}>) -> tensor<1x4x1600x2560xf16, {order = #NHWC}> {
    %c38 = arith.constant 38 : index
    %c2560 = arith.constant 2560 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x4x1600x2560xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c2560 step %c38 iter_args(%arg2 = %0) -> (tensor<1x4x1600x2560xf16, {order = #NHWC}>) {
      %map = affine.min #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 4, 1600, %map] [1, 1, 1, 1] : tensor<1x4x1600x2560xf32, {order = #NHWC}> to tensor<1x4x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
      %40 = VPU.Convert(%extracted_slice) {dstElemType = f16} : tensor<1x4x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %40 into %arg2[0, 0, 0, %arg1] [1, 4, 1600, %map] [1, 1, 1, 1] : tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x4x1600x2560xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x4x1600x2560xf16, {order = #NHWC}>
    }

    return %1 : tensor<1x4x1600x2560xf16, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 1600, 38]
    //CHECK: [[CONVERT0:%.+]] = VPU.Convert([[SLICE0]]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 38] [1, 4, 1600, 38]
    //CHECK: [[CONVERT1:%.+]] = VPU.Convert([[SLICE1]]
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONVERT0]], [[CONVERT1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 38],
    //CHECK-SAME{LITERAL}:  [0, 0, 0, 2508], [0, 0, 0, 2546]]}
}

// -----

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0034980668741113998:117>

#map1 = affine_map<(d0) -> (-d0 + 1280, 15)>
#map2 = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map3 = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1) -> (d0 * 2 - d1 + 1)>

// CHECK: func.func @UnrollConvolution([[INPUT:%.+]]: tensor<1x4x1600x2560xf16, {order = #NHWC}>)
func.func @UnrollConvolution(%arg0: tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %c15 = arith.constant 15 : index
    %c1280 = arith.constant 1280 : index
    %c0 = arith.constant 0 : index

    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x144xf16>, [#const.Reorder<#NHWC>]

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c1280 step %c15 iter_args(%arg2 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
      %size1 = affine.min #map1(%arg1)
      %offset0 = affine.max #map2(%arg1)
      %value = affine.max #map3(%arg1)
      %pad = affine.min #map4()[%value]
      %size0 = affine.apply #map5(%size1, %pad)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %offset0] [1, 4, 1600, %size0] [1, 1, 1, 1] : tensor<1x4x1600x2560xf16, {order = #NHWC}> to tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
      %padded = tensor.pad %extracted_slice low[0, 0, 1, %pad] high[0, 0, 0, 0] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC}>
      %conv = VPU.NCE.Convolution(%padded, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>, rawFilterShape = [32, 4, 3, 3], strides = [2, 2]} : tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x1x1x144xf16, {order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %conv into %arg2[0, 0, 0, %arg1] [1, 32, 800, %size1] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    }

    return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 1600, 30]
    //CHECK-NOT: tensor.pad
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]]
    //CHECK-SAME: pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 29] [1, 4, 1600, 31]
    //CHECK-NOT: tensor.pad
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]]
    //CHECK-SAME: pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 15],
    //CHECK-SAME{LITERAL}:  [0, 0, 0, 1260], [0, 0, 0, 1275]]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 1280, 28)>

!qElemType = !quant.uniform<u8:f16, 0.0034980668741113998:117>

// CHECK: func.func @UnrollEltwise([[INPUT0:%.+]]: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>,
// CHECK-SAME:                     [[INPUT1:%.+]]: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>)
func.func @UnrollEltwise(%arg0: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>, %arg1: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %c28 = arith.constant 28 : index
    %c1280 = arith.constant 1280 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %c1280 step %c28 iter_args(%arg3 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
      %size = affine.min #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %eltwise = VPU.NCE.Eltwise(%extracted_slice, %extracted_slice_1) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.380000e+02 : f64, clamp_high = 1.170000e+02 : f64, scale = 3.5136938095092773E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.380000e+02 : f64, in1_mult = [2.934300e+04], in2_mult = [2.934300e+04]>} -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %eltwise into %arg3[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    }


    return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 32, 800, 28]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 800, 28]
    //CHECK: [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]]
    //CHECK-NOT: tensor.insert_slice
    //CHECK: [[SLICE2:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 28] [1, 32, 800, 28]
    //CHECK: [[SLICE3:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 28] [1, 32, 800, 28]
    //CHECK: [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[SLICE2]], [[SLICE3]]
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[ELTWISE0]], [[ELTWISE1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 28],
    //CHECK-SAME{LITERAL}:  [0, 0, 0, 1232], [0, 0, 0, 1260]]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK: func.func @UnrollDepthToSpace([[INPUT:%.+]]: tensor<1x16x800x1280xf16, {order = #NHWC}>)
func.func @UnrollDepthToSpace(%arg0: tensor<1x16x800x1280xf16, {order = #NHWC}>) -> tensor<1x4x1600x2560xf32, {order = #NHWC}> {
    %c20 = arith.constant 20 : index
    %c1600 = arith.constant 1600 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x4x1600x2560xf32, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c1600 step %c20 iter_args(%arg2 = %0) -> (tensor<1x4x1600x2560xf32, {order = #NHWC}>) {
      %offset = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %offset, 0] [1, 16, 10, 1280] [1, 1, 1, 1] : tensor<1x16x800x1280xf16, {order = #NHWC}> to tensor<1x16x10x1280xf16, {order = #NHWC}>
      %40 = VPU.DepthToSpace(%extracted_slice) {block_size = 2 : i64, dstElemType = f32, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x10x1280xf16, {order = #NHWC}> -> tensor<1x4x20x2560xf32, {order = #NHWC}>
      %inserted_slice = tensor.insert_slice %40 into %arg2[0, 0, %arg1, 0] [1, 4, 20, 2560] [1, 1, 1, 1] : tensor<1x4x20x2560xf32, {order = #NHWC}> into tensor<1x4x1600x2560xf32, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x4x1600x2560xf32, {order = #NHWC}>
    }

    return %1 : tensor<1x4x1600x2560xf32, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 10, 1280]
    //CHECK: [[D2S0:%.+]] = VPU.DepthToSpace([[SLICE0]]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 10, 0] [1, 16, 10, 1280]
    //CHECK: [[D2S1:%.+]] = VPU.DepthToSpace([[SLICE1]]
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[D2S0]], [[D2S1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 20, 0],
    //CHECK-SAME{LITERAL}:  [0, 0, 1560, 0], [0, 0, 1580, 0]]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 254)>
#map4 = affine_map<(d0) -> (0, d0 - 358)>
#map5 = affine_map<(d0, d1) -> (-d0 - d1 + 122)>

// CHECK: func.func @Unroll2DMaxpool([[INPUT:%.+]]: tensor<1x16x512x480xf16, {order = #NHWC}>)
 func.func @Unroll2DMaxpool(%arg0: tensor<1x16x512x480xf16, {order = #NHWC}>) -> tensor<1x16x512x480xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c120 = arith.constant 120 : index
    %c256 = arith.constant 256 : index
    %c480 = arith.constant 480 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x16x512x480xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c512 step %c256 iter_args(%arg2 = %0) -> (tensor<1x16x512x480xf16, {order = #NHWC}>) {
      %2 = scf.for %arg3 = %c0 to %c480 step %c120 iter_args(%arg4 = %arg2) -> (tensor<1x16x512x480xf16, {order = #NHWC}>) {
        %3 = affine.max #map(%arg1)
        %4 = affine.max #map1(%arg1)
        %5 = affine.min #map2()[%4]
        %6 = affine.max #map3(%3)
        %7 = affine.min #map2()[%6]
        %8 = affine.max #map(%arg3)
        %9 = affine.max #map1(%arg3)
        %10 = affine.min #map2()[%9]
        %11 = affine.max #map4(%8)
        %12 = affine.min #map2()[%11]
        %13 = affine.apply #map5(%10, %12)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, %8] [1, 16, 257, %13] [1, 1, 1, 1] : tensor<1x16x512x480xf16, {order = #NHWC}> to tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
        %padded = tensor.pad %extracted_slice low[0, 0, %5, %10] high[0, 0, %7, %12] {
        ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
          tensor.yield %cst : f16
        } : tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 259, 482]> : tensor<4xsi64>, order = #NHWC}>
        %14 = VPU.NCE.MaxPool(%padded) {kernel_size = [3, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
        %cast = tensor.cast %14 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x120xf16, {order = #NHWC}>
        %inserted_slice = tensor.insert_slice %cast into %arg4[0, 0, %arg1, %arg3] [1, 16, 256, 120] [1, 1, 1, 1] : tensor<1x16x256x120xf16, {order = #NHWC}> into tensor<1x16x512x480xf16, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x512x480xf16, {order = #NHWC}>
      }
      scf.yield %2 : tensor<1x16x512x480xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x512x480xf16, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 257, 121]
    //CHECK-NOT: tensor.pad
    //CHECK: [[MAXPOOL0:%.+]] = VPU.NCE.MaxPool([[SLICE0]]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 119] [1, 16, 257, 122]
    //CHECK-NOT: tensor.pad
    //CHECK: [[MAXPOOL1:%.+]] = VPU.NCE.MaxPool([[SLICE1]]
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL0]], [[MAXPOOL1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 120], [0, 0, 0, 240], [0, 0, 0, 360], [0, 0, 256, 0], [0, 0, 256, 120], [0, 0, 256, 240], [0, 0, 256, 360]]}
}

// -----


config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> ((d0 floordiv 6) * 5 + 130)>

// CHECK: func.func @UnrollUnevenConvolution([[INPUT:%.+]]: tensor<1x96x800x1280xf16, {order = #NHWC}>)
func.func @UnrollUnevenConvolution(%arg0: tensor<1x96x800x1280xf16, {order = #NHWC}>) -> tensor<1x16x800x1280xf16, {order = #NHWC}> {
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c780 = arith.constant 780 : index
    %c800 = arith.constant 800 : index
    %c0 = arith.constant 0 : index

    %cst_0 = const.Declare tensor<16x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x96x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = tensor.empty() : tensor<1x16x800x1280xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c800 step %c6 iter_args(%arg2 = %0) -> (tensor<1x16x800x1280xf16, {order = #NHWC}>) {
      %2 = arith.cmpi ult, %arg1, %c780 : index
      %3 = arith.select %2, %c6, %c5 : index
      %4 = scf.if %2 -> (index) {
        scf.yield %arg1 : index
      } else {
        %5 = affine.apply #map(%arg1)
        scf.yield %5 : index
      }
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %4, 0] [1, 96, %3, 1280] [1, 1, 1, 1] : tensor<1x96x800x1280xf16, {order = #NHWC}> to tensor<1x96x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %42 = VPU.NCE.Convolution(%extracted_slice, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [16, 96, 1, 1], strides = [1, 1]} : tensor<1x96x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x96x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %42 into %arg2[0, 0, %4, 0] [1, 16, %3, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x800x1280xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x800x1280xf16, {order = #NHWC}>
    }


    return %1 : tensor<1x16x800x1280xf16, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 96, 6, 1280]
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 6, 0] [1, 96, 6, 1280]
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]]

    //CHECK: [[SLICEN:%.+]] = VPU.Slice [[INPUT]] [0, 0, 780, 0] [1, 96, 5, 1280]
    //CHECK: [[CONVN:%.+]] = VPU.NCE.Convolution([[SLICEN]]
    //CHECK: [[SLICEN1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 785, 0] [1, 96, 5, 1280]
    //CHECK: [[CONVN1:%.+]] = VPU.NCE.Convolution([[SLICEN1]]
    //CHECK: [[SLICEN2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 790, 0] [1, 96, 5, 1280]
    //CHECK: [[CONVN2:%.+]] = VPU.NCE.Convolution([[SLICEN2]]
    //CHECK: [[SLICEN3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 795, 0] [1, 96, 5, 1280]
    //CHECK: [[CONVN3:%.+]] = VPU.NCE.Convolution([[SLICEN3]]

    //CHECK-NOT: tensor.insert_slice
    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]],
    //CHECK-SAME:            [[CONVN]], [[CONVN1]], [[CONVN2]], [[CONVN3]])
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0]
    //CHECK-SAME{LITERAL}:  [0, 0, 780, 0], [0, 0, 785, 0], [0, 0, 790, 0], [0, 0, 795, 0]]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> ((d0 floordiv 28) * 27 + 38)>

!qElemType = !quant.uniform<u8:f16, 0.0050920921213486615:109>


// CHECK: func.func @UnrollloopWithDMAs([[INPUT:%.+]]: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>)
func.func @UnrollloopWithDMAs(%arg0: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %c27 = arith.constant 27 : index
    %c28 = arith.constant 28 : index
    %c1064 = arith.constant 1064 : index
    %c1280 = arith.constant 1280 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c1280 step %c28 iter_args(%arg2 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
      %2 = arith.cmpi ult, %arg1, %c1064 : index
      %3 = arith.select %2, %c28, %c27 : index
      %4 = scf.if %2 -> (index) {
        scf.yield %arg1 : index
      } else {
        %9 = affine.apply #map(%arg1)
        scf.yield %9 : index
      }
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %4] [1, 32, 800, %3] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg0[0, 0, 0, %4] [1, 32, 800, %3] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %5 = VPU.Copy(%extracted_slice) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
      %6 = VPU.Copy(%extracted_slice_1) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
      %7 = VPU.NCE.Eltwise(%5, %6) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.090000e+02 : f64, clamp_high = 1.460000e+02 : f64, scale = 2.3410655558109283E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.090000e+02 : f64, in1_mult = [2.863000e+04], in2_mult = [2.863000e+04]>} -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
      %8 = VPU.Copy(%7) : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %8 into %arg2[0, 0, 0, %4] [1, 32, 800, %3] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    }

    return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 800, 28]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 800, 28]
    //CHECK: [[COPY0:%.+]] = VPU.Copy([[SLICE0]])
    //CHECK: [[COPY1:%.+]] = VPU.Copy([[SLICE1]])
    //CHECK: [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[COPY0]], [[COPY1]])
    //CHECK: [[COPY2:%.+]] = VPU.Copy([[ELTWISE0]])
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[COPY2]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 28]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>
#map4 = affine_map<(d0) -> (d0 ceildiv 6)>
#map5 = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 6)>
#map6 = affine_map<(d0) -> (d0 + 2)>

//CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 ceildiv 6)>
//CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 6)>
//CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 + 2)>

!convInTiledType = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!paddedConvInTiledType = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!innerConvOutTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!convOutTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-LABEL:   @SOHConvTileOverH
// CHECK-SAME:       [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOHConvTileOverH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %c-2 = arith.constant -2 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %2 = affine.max #map(%arg1)
    %3 = affine.max #map1(%arg1)
    %4 = affine.min #map2()[%3]
    %5 = affine.max #map3(%2)
    %6 = affine.min #map2()[%5]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1]
      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    %copy_act = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
      : tensor<1x32x33x64xf16, {order = #NHWC}> -> tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %copy_weights = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
      : tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %padded = tensor.pad %copy_act low[0, 0, %4, 1] high[0, 0, %6, 1] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f16
    } : tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}> to !convInTiledType

    %dim = tensor.dim %padded, %c2 : !convInTiledType
    %7 = arith.addi %dim, %c-2 : index
    %8 = tensor.empty(%7) : !convOutTiledType
    %9 = affine.apply #map4(%7)
    %10 = scf.forall (%arg3) = (0) to (%7) step (%9) shared_outs(%arg4 = %8) -> (!convOutTiledType) {
      %11 = affine.min #map5(%arg3, %7)[%7]
      %12 = affine.apply #map6(%11)
      %extracted_slice_1 = tensor.extract_slice %padded[0, 0, %arg3, 0] [1, 32, %12, 66] [1, 1, 1, 1]
        : !convInTiledType to !paddedConvInTiledType
      %13 = VPU.NCE.Convolution(%extracted_slice_1, %copy_weights) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
        : !paddedConvInTiledType, tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> !innerConvOutTiledType
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %13 into %arg4[0, 0, %arg3, 0] [1, 256, %11, 64] [1, 1, 1, 1]
          : !innerConvOutTiledType into !convOutTiledType
      }
    }

    %cast = tensor.cast %10
      : !convOutTiledType to tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %copy_output = VPU.Copy(%cast) {out_mem_space = @DDR}
      : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>

    %inserted_slice = tensor.insert_slice %copy_output into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1]
      : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK-DAG:   [[CST0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[CST1:%.+]] = arith.constant 1 : index
// CHECK-DAG:   [[CST2:%.+]] = arith.constant 2 : index
// CHECK-DAG:   [[CST_MIN2:%.+]] = arith.constant -2 : index

// CHECK:       [[TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
// CHECK-SAME:    tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

// CHECK:       [[COPY_TILE0:%.+]] = VPU.Copy([[TILE0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x33x64xf16, {order = #NHWC}> -> tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[PAD_TILE_0:%.+]] = tensor.pad [[COPY_TILE0]] low[0, 0, [[CST1]], 1] high[0, 0, [[CST0]], 1]
// CHECK:         : tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:    to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[DIM_H0:%.+]] = tensor.dim [[PAD_TILE_0]], [[CST2]]
// CHECK:       [[LOOP_END0:%.+]] = arith.addi [[DIM_H0]], [[CST_MIN2]] : index
// CHECK:       [[STEP0:%.+]] = affine.apply #[[$MAP]]([[LOOP_END0]])
// CHECK:       [[OUT_BUFF0:%.+]] = tensor.empty() : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[CONV0:%.+]] = scf.forall ([[ITER0:%.+]]) = (0) to ([[LOOP_END0]]) step ([[STEP0]])
// CHECK-SAME:    shared_outs([[SHARED_OUT0:%.+]] = [[OUT_BUFF0]])
// CHECK-SAME:    -> (tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
// CHECK:         [[OUT_SZ0:%.+]] = affine.min #[[$MAP1]]([[ITER0]], [[LOOP_END0]])[[[LOOP_END0]]]

// CHECK:         [[IN_MC0:%.+]] = tensor.extract_slice [[PAD_TILE_0]][0, 0, [[ITER0]], 0] [1, 32, {{%.+}}, 66]
// CHECK:         [[INNER_CONV0:%.+]] = VPU.NCE.Convolution([[IN_MC0]], {{%.+}})
// CHECK-SAME:      : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice [[INNER_CONV0]] into [[SHARED_OUT0]][0, 0, [[ITER0]], 0] [1, 256, [[OUT_SZ0]], 64]
// CHECK-SAME:        : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:        into tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         }

// CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[CONV0]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:       [[TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
// CHECK-SAME:    tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

// CHECK:       [[COPY_TILE1:%.+]] = VPU.Copy([[TILE1]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x33x64xf16, {order = #NHWC}> -> tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[PAD_TILE_1:%.+]] = tensor.pad [[COPY_TILE1]] low[0, 0, [[CST0]], 1] high[0, 0, [[CST1]], 1]
// CHECK:         : tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:    to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[OUT_BUFF1:%.+]] = tensor.empty() : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[CONV1:%.+]] = scf.forall ([[ITER1:%.+]]) = (0) to ({{[^:]+}}) step ({{[^:]+}}) shared_outs([[SHARED_OUT1:%.+]] = [[OUT_BUFF1]])
// CHECK-SAME:    -> (tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

// CHECK:         [[IN_MC1:%.+]] = tensor.extract_slice [[PAD_TILE_1]][0, 0, [[ITER1]], 0] [1, 32, {{[^:]+}}, 66]
// CHECK:         [[INNER_CONV1:%.+]] = VPU.NCE.Convolution([[IN_MC1]], {{[^:]+}})
// CHECK-SAME:      : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice [[INNER_CONV1]] into [[SHARED_OUT1]][0, 0, [[ITER1]], 0] [1, 256, {{[^:]+}}, 64]
// CHECK-SAME:        : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:        into tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         }

// CHECK:       [[COPY_OUT_1:%.+]] = VPU.Copy([[CONV1]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:       VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 32, 0{{\]\]}}}
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>
#map4 = affine_map<(d0) -> (-d0 + 256, 96)>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (-d0 + 256, 96)>

!convInTiledType = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!convOutTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!innerConvOutTiledType = tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!castedConvOutTiledType = tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL:   @SOKConvTileOverH
// CHECK-SAME:       [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKConvTileOverH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %c-2 = arith.constant -2 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %2 = affine.max #map(%arg1)
    %3 = affine.max #map1(%arg1)
    %4 = affine.min #map2()[%3]
    %5 = affine.max #map3(%2)
    %6 = affine.min #map2()[%5]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1]
      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    %copy_act = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
      : tensor<1x32x33x64xf16, {order = #NHWC}> -> tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %copy_weights = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
      : tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %padded = tensor.pad %copy_act low[0, 0, %4, 1] high[0, 0, %6, 1] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f16
    } : tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}> to !convInTiledType

    %dim = tensor.dim %padded, %c2 : !convInTiledType
    %7 = arith.addi %dim, %c-2 : index
    %8 = tensor.empty(%7) : !convOutTiledType
    %9 = scf.forall (%arg3) = (0) to (256) step (96) shared_outs(%arg4 = %8) -> (!convOutTiledType) {
      %10 = affine.min #map4(%arg3)
      %extracted_slice_1 = tensor.extract_slice %padded[0, 0, 0, 0] [1, 32, %dim, 66] [1, 1, 1, 1]
        : !convInTiledType to !convInTiledType
      %extracted_slice_2 = tensor.extract_slice %copy_weights[%arg3, 0, 0, 0] [%10, 32, 3, 3] [1, 1, 1, 1]
        : tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        to tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
      %11 = VPU.NCE.Convolution(%extracted_slice_1, %extracted_slice_2) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 32, 3, 3], strides = [1, 1]
      } : !convInTiledType, tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
      -> !innerConvOutTiledType
      %cast_3 = tensor.cast %11
        : !innerConvOutTiledType to !castedConvOutTiledType
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %cast_3 into %arg4[0, %arg3, 0, 0] [1, %10, 64, 64] [1, 1, 1, 1]
          : !castedConvOutTiledType into !convOutTiledType
      }
    }
    %cast = tensor.cast %9 : !convOutTiledType to tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %copy_output = VPU.Copy(%cast) {out_mem_space = @DDR}
      : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>

    %inserted_slice = tensor.insert_slice %copy_output into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1]
      : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK:       [[TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
// CHECK-SAME:    tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK:       [[COPY_TILE0:%.+]] = VPU.Copy([[TILE0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x33x64xf16, {order = #NHWC}> -> tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[COPY_W_TILE0:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[OUT_BUFF0:%.+]] = tensor.empty() : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[CONV0:%.+]] = scf.forall ([[ITER0:%.+]]) = (0) to (256) step (96)
// CHECK-SAME:    shared_outs([[SHARED_OUT0:%.+]] = [[OUT_BUFF0]])
// CHECK-SAME:    -> (tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
// CHECK:         [[OUT_SZ0:%.+]] = affine.min #[[$MAP]]([[ITER0]])

// CHECK:         [[WEIGHTS0:%.+]] = tensor.extract_slice [[COPY_W_TILE0]][[[ITER0]], 0, 0, 0] [[[OUT_SZ0]], 32, 3, 3]
// CHECK:         [[INNER_CONV0:%.+]] = VPU.NCE.Convolution([[COPY_TILE0]], [[WEIGHTS0]])
// CHECK-SAME:      : tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      -> tensor<1x?x32x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice [[INNER_CONV0]] into [[SHARED_OUT0]][0, [[ITER0]], 0, 0] [1, [[OUT_SZ0]], 32, 64]
// CHECK-SAME:        : tensor<1x?x32x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:        into tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         }

// CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[CONV0]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:       [[TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
// CHECK-SAME:    tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK:       [[COPY_TILE1:%.+]] = VPU.Copy([[TILE1]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x33x64xf16, {order = #NHWC}> -> tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[COPY_W_TILE1:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[OUT_BUFF1:%.+]] = tensor.empty() : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[CONV1:%.+]] = scf.forall ([[ITER1:%.+]]) = (0) to (256) step (96) shared_outs([[SHARED_OUT1:%.+]] = [[OUT_BUFF1]])
// CHECK-SAME:    -> (tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
// CHECK:         [[OUT_SZ1:%.+]] = affine.min #[[$MAP]]([[ITER1]])

// CHECK:         [[WEIGHTS1:%.+]] = tensor.extract_slice [[COPY_W_TILE1]][[[ITER1]], 0, 0, 0] [[[OUT_SZ1]], 32, 3, 3]
// CHECK:         [[INNER_CONV1:%.+]] = VPU.NCE.Convolution([[COPY_TILE1]], [[WEIGHTS1]])
// CHECK-SAME:      : tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      -> tensor<1x?x32x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice [[INNER_CONV1]] into [[SHARED_OUT1]][0, [[ITER1]], 0, 0] [1, [[OUT_SZ1]], 32, 64]
// CHECK-SAME:        : tensor<1x?x32x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:        into tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         }

// CHECK:       [[COPY_OUT_1:%.+]] = VPU.Copy([[CONV1]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:       VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 32, 0{{\]\]}}}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>
#map1 = affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 126)>
#map4 = affine_map<(d0) -> (-d0 + 160, 54)>
#map5 = affine_map<(d0) -> (d0 + 2)>
#map6 = affine_map<(d0) -> (-d0 + 320, 107)>
#map7 = affine_map<(d0) -> (d0 floordiv 2)>

!inputConvType = tensor<1x32x160x256xf16, {order = #NHWC}>
!outputD2SType = tensor<1x4x320x512xf16, {order = #NHWC}>
!inD2SType = tensor<1x16x160x128xf16, {mem_space = @CMX_NN, order = #NHWC}>

!inputConvTiledType = tensor<1x32x160x129xf16, {mem_space = @CMX_NN, order = #NHWC}>
!inputConvTiledDDRType = tensor<1x32x160x129xf16, {order = #NHWC}>
!inputConvTiledPaddedType = tensor<1x32x162x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 162, 258]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!outConvTiledType = tensor<1x16x160x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 160, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!outD2STiledType = tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}>
!outD2STiledDDRType = tensor<1x4x320x256xf16, {order = #NHWC}>

!innerConvInputType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 56, 258]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!innerConvOutputType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!innerConvOutCastedType = tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!innerD2SInType = tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 160, 128]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!innerD2SOutType = tensor<1x4x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 320, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (-d0 + 160, 54)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (-d0 + 320, 107)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL: @ChainConvAddD2SWithMC
// CHECK-SAME:       [[INPUT:%arg[0-9]]]: tensor<1x32x160x256xf16, {order = #NHWC}>
func.func @ChainConvAddD2SWithMC(%arg0: !inputConvType) -> !outputD2SType {
  %c-2 = arith.constant -2 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %cst_0 = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    : tensor<16x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

  %0 = tensor.empty() : !outputD2SType
  %1 = scf.for %arg1 = %c0 to %c512 step %c256 iter_args(%arg2 = %0) -> (!outputD2SType) {
    %2 = affine.max #map(%arg1)
    %3 = affine.max #map1(%arg1)
    %4 = affine.min #map2()[%3]
    %5 = affine.max #map3(%2)
    %6 = affine.min #map2()[%5]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %2] [1, 32, 160, 129] [1, 1, 1, 1]
      : !inputConvType to !inputConvTiledDDRType

    %copy_act = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
      : !inputConvTiledDDRType -> !inputConvTiledType
    %copy_weights = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
      : tensor<16x32x3x3xf16, {order = #NHWC}> -> tensor<16x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %padded = tensor.pad %copy_act low[0, 0, 1, %4] high[0, 0, 1, %6] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f16
    } : !inputConvTiledType to !inputConvTiledPaddedType
    %dim = tensor.dim %padded, %c3 : !inputConvTiledPaddedType
    %7 = arith.addi %dim, %c-2 : index
    %8 = tensor.empty(%7) : !outConvTiledType
    %9 = scf.forall (%arg3) = (0) to (160) step (54) shared_outs(%arg4 = %8) -> (!outConvTiledType) {
      %12 = affine.min #map4(%arg3)
      %13 = affine.apply #map5(%12)
      %extracted_slice_1 = tensor.extract_slice %padded[0, 0, %arg3, 0] [1, 32, %13, %dim] [1, 1, 1, 1]
        : !inputConvTiledPaddedType to !innerConvInputType
      %14 = VPU.NCE.Convolution(%extracted_slice_1, %copy_weights) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, rawFilterShape = [16, 32, 3, 3], strides = [1, 1]
      } : !innerConvInputType, tensor<16x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !innerConvOutputType
      %cast_2 = tensor.cast %14 : !innerConvOutputType to !innerConvOutCastedType
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %cast_2 into %arg4[0, 0, %arg3, 0] [1, 16, %12, 256] [1, 1, 1, 1]
          : !innerConvOutCastedType into !outConvTiledType
      }
    }
    %cast = tensor.cast %9 : !outConvTiledType to !inD2SType
    %10 = tensor.empty() : !outD2STiledType
    %11 = scf.forall (%arg3) = (0) to (320) step (107) shared_outs(%arg4 = %10)
        -> (!outD2STiledType) {
      %12 = affine.min #map6(%arg3)
      %13 = affine.apply #map7(%arg3)
      %14 = affine.apply #map7(%12)
      %extracted_slice_1 = tensor.extract_slice %cast[0, 0, %13, 0] [1, 16, %14, 128] [1, 1, 1, 1]
        : !inD2SType to !innerD2SInType
      %15 = VPU.DepthToSpace(%extracted_slice_1)
        {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
        : !innerD2SInType -> !innerD2SOutType
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %15 into %arg4[0, 0, %arg3, 0] [1, 4, %12, 256] [1, 1, 1, 1]
          : !innerD2SOutType into !outD2STiledType
      }
    }

    %copy_output = VPU.Copy(%11) {out_mem_space = @DDR}
      : !outD2STiledType -> !outD2STiledDDRType
    %inserted_slice = tensor.insert_slice %copy_output into %arg2[0, 0, 0, %arg1] [1, 4, 320, 256] [1, 1, 1, 1]
      : !outD2STiledDDRType into !outputD2SType
    scf.yield %inserted_slice : !outputD2SType
  }
  return %1 : !outputD2SType

// CHECK-DAG:   [[CST0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[CST1:%.+]] = arith.constant 1 : index
// CHECK-DAG:   [[CST3:%.+]] = arith.constant 3 : index

// CHECK:       [[TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 160, 129]
// CHECK-SAME:    : tensor<1x32x160x256xf16, {order = #NHWC}> to tensor<1x32x160x129xf16, {order = #NHWC}>

// CHECK:       [[COPY_TILE0:%.+]] = VPU.Copy([[TILE0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x160x129xf16, {order = #NHWC}> -> tensor<1x32x160x129xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[PAD_TILE_0:%.+]] = tensor.pad [[COPY_TILE0]] low[0, 0, 1, [[CST1]]] high[0, 0, 1, [[CST0]]]
// CHECK:         : tensor<1x32x160x129xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:    to tensor<1x32x162x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 162, 258]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[DIM_W0:%.+]] = tensor.dim [[PAD_TILE_0]], [[CST3]]
// CHECK:       [[OUT_CONV_BUFF0:%.+]] = tensor.empty() : tensor<1x16x160x128xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[CONV0:%.+]] = scf.forall ([[ITER0:%.+]]) = (0) to (160) step (54)
// CHECK-SAME:    shared_outs([[SHARED_OUT0:%.+]] = [[OUT_CONV_BUFF0]])
// CHECK-SAME:    -> (tensor<1x16x160x128xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

// CHECK:         [[OUT_SZ0:%.+]] = affine.min #[[$MAP]]([[ITER0]])
// CHECK:         [[IN_SZ0:%.+]] = affine.apply #[[$MAP1]]([[OUT_SZ0]])

// CHECK:         [[INNER_MC_INPUT0:%.+]] = tensor.extract_slice [[PAD_TILE_0]][0, 0, [[ITER0]], 0] [1, 32, [[IN_SZ0]], [[DIM_W0]]]
// CHECK:         [[INNER_CONV0:%.+]] = VPU.NCE.Convolution([[INNER_MC_INPUT0]], {{[^:]+}})
// CHECK-SAME:      : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 56, 258]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:      -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:         [[CAST0:%.+]] = tensor.cast [[INNER_CONV0]]
// CHECK-SAME:       : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:       to tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 128]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice [[CAST0]] into [[SHARED_OUT0]][0, 0, [[ITER0]], 0] [1, 16, [[OUT_SZ0]], 128]
// CHECK-SAME:        : tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 128]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:        into tensor<1x16x160x128xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         }

// CHECK:       [[OUT_D2S_BUFF0:%.+]] = tensor.empty() : tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[D2S0:%.+]] = scf.forall ([[ITER_D2S0:%.+]]) = (0) to (320) step (107)
// CHECK-SAME:    shared_outs([[SHARED_OUT_D2S_0:%.+]] = [[OUT_D2S_BUFF0]])
// CHECK-SAME:    -> (tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

// CHECK:          [[INNER_D2S_INPUT0:%.+]] = tensor.extract_slice [[CONV0]][0, 0, {{[^:]+}}, 0] [1, 16, {{[^:]+}}, 128]
// CHECK-SAME:       : tensor<1x16x160x128xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:       to tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 160, 128]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:          [[INNER_D2S:%.+]] = VPU.DepthToSpace([[INNER_D2S_INPUT0]])
// CHECK-SAME:       : tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 160, 128]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:       -> tensor<1x4x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 320, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK:          scf.forall.in_parallel
// CHECK:            tensor.parallel_insert_slice [[INNER_D2S]] into [[SHARED_OUT_D2S_0]][0, 0, [[ITER_D2S0]], 0] [1, 4, {{[^:]+}}, 256]
// CHECK-SAME:         : tensor<1x4x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 320, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:         into tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[D2S0]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x4x320x256xf16, {order = #NHWC}>

// CHECK:       [[TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 127] [1, 32, 160, 129]
// CHECK-SAME:    : tensor<1x32x160x256xf16, {order = #NHWC}> to tensor<1x32x160x129xf16, {order = #NHWC}>

// CHECK:       [[COPY_TILE1:%.+]] = VPU.Copy([[TILE1]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x160x129xf16, {order = #NHWC}> -> tensor<1x32x160x129xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[PAD_TILE_1:%.+]] = tensor.pad [[COPY_TILE1]] low[0, 0, 1, [[CST0]]] high[0, 0, 1, [[CST1]]]

// CHECK:       [[OUT_CONV_BUFF1:%.+]] = tensor.empty() : tensor<1x16x160x128xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[CONV1:%.+]] = scf.forall ([[ITER_CONV1:%.+]]) = (0) to (160) step (54) shared_outs([[SHARED_OUT1:%.+]] = [[OUT_CONV_BUFF1]])
// CHECK-SAME:    -> (tensor<1x16x160x128xf16, {mem_space = @CMX_NN, order = #NHWC}>)

// CHECK:         [[INNNER_CONV_INPUT1:%.+]] = tensor.extract_slice [[PAD_TILE_1]][0, 0, [[ITER_CONV1]], 0] [1, 32, {{[^:]+}}, {{[^:]+}}]
// CHECK:         [[INNER_CONV1:%.+]] = VPU.NCE.Convolution([[INNNER_CONV_INPUT1]], {{%.+}})
// CHECK-SAME:      : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 56, 258]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:      -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:         [[CAST1:%.+]] = tensor.cast [[INNER_CONV1]]
// CHECK-SAME:      : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 128]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CAST1]] into [[SHARED_OUT1]][0, 0, [[ITER_CONV1]], 0] [1, 16, {{[^:]+}}, 128]
// CHECK-SAME:        : tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 128]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:        into tensor<1x16x160x128xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[OUT_D2S_BUFF1:%.+]] = tensor.empty() : tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[D2S1:%.+]] = scf.forall ([[ITER_D2S1:%.+]]) = (0) to (320) step (107)
// CHECK-SAME:    shared_outs([[SHARED_OUT_D2S_1:%.+]] = [[OUT_D2S_BUFF1]])
// CHECK-SAME:    -> (tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

// CHECK:          [[INNER_D2S_INPUT1:%.+]] = tensor.extract_slice [[CONV1]][0, 0, {{[^:]+}}, 0] [1, 16, {{[^:]+}}, 128]
// CHECK:          [[INNER_D2S_1:%.+]] = VPU.DepthToSpace([[INNER_D2S_INPUT1]])
// CHECK-SAME:       : tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 160, 128]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:       -> tensor<1x4x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 320, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:          scf.forall.in_parallel
// CHECK:            tensor.parallel_insert_slice [[INNER_D2S_1]] into [[SHARED_OUT_D2S_1]][0, 0, [[ITER_D2S1]], 0] [1, 4, {{[^:]+}}, 256]
// CHECK-SAME:         : tensor<1x4x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 320, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:         into tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[COPY_OUT_1:%.+]] = VPU.Copy([[D2S1]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x4x320x256xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x4x320x256xf16, {order = #NHWC}>

// CHECK:       VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 256{{\]\]}}}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>
#map4 = affine_map<(d0) -> (d0 ceildiv 3)>
#map5 = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 3)>
#map6 = affine_map<(d0) -> (d0 + 2)>

!paddedInputTiledType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!outputTiledType = tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!innerInputTiledType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 24, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!innerOutputTiledType = tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!castedOutputTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 ceildiv 3)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 3)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 + 2)>

// CHECK-LABEL: @TwoAxisTilingNCEConvSOH
// CHECK-SAME:       [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @TwoAxisTilingNCEConvSOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %c3 = arith.constant 3 : index
  %c-2 = arith.constant -2 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %2 = scf.for %arg3 = %c0 to %c64 step %c32 iter_args(%arg4 = %arg2) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %3 = affine.max #map(%arg1)
      %4 = affine.max #map1(%arg1)
      %5 = affine.min #map2()[%4]
      %6 = affine.max #map3(%3)
      %7 = affine.min #map2()[%6]
      %8 = affine.max #map(%arg3)
      %9 = affine.max #map1(%arg3)
      %10 = affine.min #map2()[%9]
      %11 = affine.max #map3(%8)
      %12 = affine.min #map2()[%11]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, %8] [1, 32, 33, 33] [1, 1, 1, 1]
        : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x33xf16, {order = #NHWC}>

      %copy_act = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
        : tensor<1x32x33x33xf16, {order = #NHWC}> -> tensor<1x32x33x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
      %copy_weights = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
        : tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

      %padded = tensor.pad %copy_act low[0, 0, %5, %10] high[0, 0, %7, %12] {
      ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
        tensor.yield %cst : f16
      } : tensor<1x32x33x33xf16, {mem_space = @CMX_NN, order = #NHWC}> to !paddedInputTiledType

      %dim = tensor.dim %padded, %c2 : !paddedInputTiledType
      %13 = arith.addi %dim, %c-2 : index
      %dim_1 = tensor.dim %padded, %c3 : !paddedInputTiledType
      %14 = arith.addi %dim_1, %c-2 : index

      %15 = tensor.empty(%13, %14) : !outputTiledType
      %16 = affine.apply #map4(%13)
      %17 = scf.forall (%arg5) = (0) to (%13) step (%16) shared_outs(%arg6 = %15) -> (!outputTiledType) {
        %18 = affine.min #map5(%arg5, %13)[%13]
        %19 = affine.apply #map6(%18)
        %extracted_slice_2 = tensor.extract_slice %padded[0, 0, %arg5, 0] [1, 32, %19, %dim_1] [1, 1, 1, 1]
          : !paddedInputTiledType to !innerInputTiledType
        %20 = VPU.NCE.Convolution(%extracted_slice_2, %copy_weights) {
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3],
          strides = [1, 1], tiling_index = 0 : i64
        } : !innerInputTiledType, tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> !innerOutputTiledType
        %cast_3 = tensor.cast %20 : !innerOutputTiledType to !castedOutputTiledType
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %cast_3 into %arg6[0, 0, %arg5, 0] [1, 256, %18, 64] [1, 1, 1, 1]
            : !castedOutputTiledType into !outputTiledType
        }
      }
      %cast = tensor.cast %17 : !outputTiledType to tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
      %copy_output = VPU.Copy(%cast) {out_mem_space = @DDR}
      : tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x32xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %copy_output into %arg4[0, 0, %arg1, %arg3] [1, 256, 32, 32] [1, 1, 1, 1]
        : tensor<1x256x32x32xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    scf.yield %2 : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK-DAG:   [[CST0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[CST1:%.+]] = arith.constant 1 : index
// CHECK-DAG:   [[CST2:%.+]] = arith.constant 2 : index
// CHECK-DAG:   [[CST3:%.+]] = arith.constant 3 : index

// CHECK:       [[TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 33]
// CHECK-SAME:    : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x33xf16, {order = #NHWC}>
// CHECK:       [[COPY_TILE0:%.+]] = VPU.Copy([[TILE_0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x33x33xf16, {order = #NHWC}> -> tensor<1x32x33x33xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[PAD0:%.+]] = tensor.pad [[COPY_TILE0:%.+]] low[0, 0, [[CST1]], [[CST1]]] high[0, 0, [[CST0]], [[CST0]]]
// CHECK:       : tensor<1x32x33x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:  to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[OUT_BUFF_0:%.+]] = tensor.empty() : tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[CONV0:%.+]] = scf.forall ([[ITER_0:%.+]]) = (0) to ({{%[0-9]+}}) step ({{%[0-9]+}})
// CHECK-SAME:    shared_outs([[SHARED_OUT_0:%.+]] = [[OUT_BUFF_0]]) -> (tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)

// CHECK:         [[INNER_CONV_INPUT_0:%.+]] = tensor.extract_slice [[PAD0]][0, 0, [[ITER_0]], 0] [1, 32, {{[^:]+}}, {{[^:]+}}]
// CHECK-SAME:      : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:      to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 24, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:         [[INNER_CONV_0:%.+]] = VPU.NCE.Convolution([[INNER_CONV_INPUT_0]], {{[^:]+}})
// CHECK-SAME:      : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 24, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:      -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:         [[CAST_0:%.+]] = tensor.cast [[INNER_CONV_0]]
// CHECK-SAME:      : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:      to tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 32]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CAST_0]] into [[SHARED_OUT_0]][0, 0, [[ITER_0]], 0] [1, 256, {{[^:]+}}, 32]
// CHECK-SAME:        : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 32]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:        into tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[CONV0]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x32xf16, {order = #NHWC}>

// CHECK:       [[TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 31] [1, 32, 33, 33]
// CHECK:       [[COPY_TILE1:%.+]] = VPU.Copy([[TILE_1]]) {out_mem_space = @CMX_NN}
// CHECK:       [[PAD1:%.+]] = tensor.pad [[COPY_TILE1]] low[0, 0, [[CST1]], [[CST0]]] high[0, 0, [[CST0]], [[CST1]]] {

// CHECK:       [[OUT_BUFF_1:%.+]] = tensor.empty() : tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:       [[CONV1:%.+]] = scf.forall ([[ITER_1:%.+]]) = (0) to ({{%[0-9]+}}) step ({{%[0-9]+}})
// CHECK-SAME:    shared_outs([[SHARED_OUT_1:%.+]] = [[OUT_BUFF_1]]) -> (tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)

// CHECK:         [[INNER_CONV_INPUT_1:%.+]] = tensor.extract_slice [[PAD1]][0, 0, [[ITER_1]], 0] [1, 32, {{[^:]+}}, {{[^:]+}}]
// CHECK:         [[INNER_CONV_1:%.+]] = VPU.NCE.Convolution([[INNER_CONV_INPUT_1]], {{[^:]+}})
// CHECK-SAME:      -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         [[CAST_1:%.+]] = tensor.cast [[INNER_CONV_1]]
// CHECK-SAME:      : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:      to tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 32]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CAST_1]] into [[SHARED_OUT_1]][0, 0, [[ITER_1]], 0] [1, 256, {{[^:]+}}, 32]
// CHECK:             : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 32]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:        into tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[COPY_OUT_1:%.+]] = VPU.Copy([[CONV1]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x32xf16, {order = #NHWC}>

// CHECK:       [[TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 33]
// CHECK:       [[COPY_TILE2:%.+]] = VPU.Copy([[TILE_2]]) {out_mem_space = @CMX_NN}
// CHECK:       [[CONV2:%.+]] = scf.forall ({{[^:]+}}) = (0) to ({{%[0-9]+}}) step ({{%[0-9]+}})
// CHECK-SAME:    -> (tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK:         [[INNER_CONV_2:%.+]] = VPU.NCE.Convolution

// CHECK:       [[COPY_OUT_2:%.+]] = VPU.Copy([[CONV2]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x32xf16, {order = #NHWC}>

// CHECK:       [[TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 31] [1, 32, 33, 33]
// CHECK:       [[COPY_TILE3:%.+]] = VPU.Copy([[TILE_3]]) {out_mem_space = @CMX_NN}
// CHECK:       [[CONV3:%.+]] = scf.forall ({{[^:]+}}) = (0) to ({{%[0-9]+}}) step ({{%[0-9]+}})
// CHECK-SAME:    -> (tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK:         [[INNER_CONV_3:%.+]] = VPU.NCE.Convolution

// CHECK:       [[COPY_OUT_3:%.+]] = VPU.Copy([[CONV3]]) {out_mem_space = @DDR}
// CHECK-SAME:    : tensor<1x256x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x32xf16, {order = #NHWC}>

// CHECK:       VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]], [[COPY_OUT_2]], [[COPY_OUT_3]])
// CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32], [0, 0, 32, 0], [0, 0, 32, 32]]}
}
