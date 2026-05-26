//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true enable-is-reduce-supported" --scf-vertical-fusion --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (-d0 + 512, 112)>

// CHECK-LABEL: @MergeNCEReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeNCEReduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 3]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.NCE.Reduce(%0) {
     axes = [1], input_padding = [0, 12, 0, 0],
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <NOOP>,
     clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 5]
   } -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

   // CHECK-DAG:    [[C112:%.+]] = arith.constant 112 : index
   // CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
   // CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
   // CHECK-DAG:    [[EMPT:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK:        [[LOOP:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C112]] iter_args([[ARG2:%.+]] = [[EMPT]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
   // CHECK-NEXT:      [[MIN:%.+]] = affine.min #[[$MAP1]]([[ITER]])
   // CHECK-NEXT:      [[IN_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, [[MIN]]] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NCHW}>
   // CHECK-NEXT:      [[PERMUTE:%.+]] = VPU.NCE.Permute([[IN_SLICE]])
   // CHECK-SAME:           -> tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK-NEXT:      [[REDUCE:%.+]] = VPU.NCE.Reduce([[PERMUTE]])
   // CHECK-SAME:           axes = [1]
   // CHECK-SAME:           input_padding = [0, 12, 0, 0]
   // CHECK-SAME:           -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[ARG2]][0, 0, 0, [[ITER]]] [1, 1, 175, [[MIN]]] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK-NEXT:   }
   // CHECK-NEXT:   return [[LOOP]] : tensor<1x1x175x512xf16, {order = #NHWC}>
}
