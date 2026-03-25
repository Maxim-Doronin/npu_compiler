//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true enable-is-reduce-supported" --apply-tiling="enable-scf-tiling=true" --cse --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> ((d0 floordiv 103) * 102 + 2)>

// CHECK-LABEL: @ApplyTilingNCEReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16, {order = #NHWC}>
func.func @ApplyTilingNCEReduce(%arg0 : tensor<1x32x175x512xf16, {order = #NHWC}>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Reduce(%arg0) {
     axes = [1], input_padding = [0, 12, 0, 0],
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <NOOP>,
     clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 5]
   } -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %0 : tensor<1x1x175x512xf16, {order = #NHWC}>

   // CHECK-DAG:  [[EMPT:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK-DAG:  [[C206:%.+]] = arith.constant 206 : index
   // CHECK-DAG:  [[C102:%.+]] = arith.constant 102 : index
   // CHECK-DAG:  [[C0:%.+]] = arith.constant 0 : index
   // CHECK-DAG:  [[C512:%.+]] = arith.constant 512 : index
   // CHECK-DAG:  [[C103:%.+]] = arith.constant 103 : index
   // CHECK:      [[REDUCED:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C103]] iter_args([[ARG2:%.+]] = [[EMPT]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
   // CHECK:         [[CMP:%.+]] = arith.cmpi ult, [[ITER]], [[C206]] : index
   // CHECK:         [[SIZE:%.+]] = arith.select [[CMP]], [[C103]], [[C102]] : index
   // CHECK:         [[OFFSET:%.+]] = scf.if [[CMP]] -> (index) {
   // CHECK:            scf.yield [[ITER]] : index
   // CHECK:         } else {
   // CHECK:            [[ADJOFF:%.+]] = affine.apply #[[$MAP1]]([[ITER]])
   // CHECK:            scf.yield [[ADJOFF]] : index
   // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[OFFSET]]] [1, 32, 175, [[SIZE]]] [1, 1, 1, 1] : tensor<1x32x175x512xf16, {order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK:         [[REDUCED_SLICE:%.+]] = VPU.NCE.Reduce([[SLICE]])
   // CHECK:         [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCED_SLICE]] into [[ARG2]][0, 0, 0, [[OFFSET]]] [1, 1, 175, [[SIZE]]] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK:         scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK-NEXT: }
   // CHECK:      return [[REDUCED]] : tensor<1x1x175x512xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK-LABEL: @ApplyTilingDynamicNCEReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingDynamicNCEReduce(%arg0 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
   %0 = VPU.NCE.Reduce(%arg0) {
     axes = [1], input_padding = [0, 12, 0, 0],
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <NOOP>,
     clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 5]
   } -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
   return %0 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[ARG2:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[SIZE:%.+]] = affine.min #[[$MAP1]]([[IDX]])[[[DIM]]]
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[SIZE]]] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.NCE.Reduce([[SLICE]])
// CHECK-SAME:          -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into %arg2[0, 0, 0, [[IDX]]] [1, 1, 175, [[SIZE]]] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:       scf.yield [[INSERT]] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
}
