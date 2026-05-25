//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --full-unroll-scf-loop %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 112, 32)>
#map1 = affine_map<(d0) -> (((d0 + 15) floordiv 16) * 16)>

!actType = tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!actTypeDDR = tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, order = #NHWC}>
!outType = tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHCW}>
!outTypeDDR = tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, order = #NHCW}>

// CHECK-LABEL: @NCEEltwiseSOK
// CHECK-SAME:       [[INPUT0:%[^:]+]]: tensor<1x112x12x12xf16, {order = #NHWC}>
// CHECK-SAME:       [[INPUT1:%[^:]+]]: tensor<1x112x12x12xf16, {order = #NHWC}>
func.func @NCEEltwiseSOK(%arg0: tensor<1x112x12x12xf16, {order = #NHWC}>, %arg1: tensor<1x112x12x12xf16, {order = #NHWC}>) -> tensor<1x112x12x12xf16, {order = #NHCW}> {
  %0 = tensor.empty() : tensor<1x112x12x12xf16, {order = #NHCW}>
  %1 = scf.forall (%arg2) = (0) to (112) step (32) shared_outs(%arg3 = %0) -> (tensor<1x112x12x12xf16, {order = #NHCW}>) {
    %2 = affine.min #map(%arg2)
    %3 = affine.apply #map1(%2)

    %extracted_slice = tensor.extract_slice %arg0[0, %arg2, 0, 0] [1, %3, 12, 12] [1, 1, 1, 1]
        : tensor<1x112x12x12xf16, {order = #NHWC}> to !actTypeDDR
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg2, 0, 0] [1, %3, 12, 12] [1, 1, 1, 1]
        : tensor<1x112x12x12xf16, {order = #NHWC}> to !actTypeDDR

    %4 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN} : !actTypeDDR -> !actType
    %5 = VPU.Copy(%extracted_slice_0) {out_mem_space = @CMX_NN} : !actTypeDDR -> !actType

    %6 = VPU.NCE.Eltwise(%4, %5) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
        -> !outType

    %7 = VPU.Copy(%6) : !outType -> !outTypeDDR
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %7 into %arg3[0, %arg2, 0, 0] [1, %2, 12, 12] [1, 1, 1, 1]
          : !outTypeDDR into tensor<1x112x12x12xf16, {order = #NHCW}>
    }
  }
  return %1 : tensor<1x112x12x12xf16, {order = #NHCW}>

    // TODO:  E#192457 - scf.forall loop shows SEGMENTED input; however, after unroll we have DUPLICATED distribution.
    //        That happens due to scf MC currently borrowing the tiling logic. This should be fixed when proper
    //        multiclustering is implemented.

    // CHECK:        [[IN0_COPY:%.+]] = VPU.Copy([[INPUT0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<1x112x12x12xf16, {order = #NHWC}>
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x112x12x12xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[IN1_COPY:%.+]] = VPU.Copy([[INPUT1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:          : tensor<1x112x12x12xf16, {order = #NHWC}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x112x12x12xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN0_COPY]], [[IN1_COPY]])
    // CHECK-SAME:           {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x112x12x12xf16, #NHCW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1],
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[OUT_COPY:%.+]] = VPU.Copy([[ELTWISE]])
    // CHECK-SAME:       : !VPU.DistributedTensor<1x112x12x12xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]
    // CHECK-SAME:       -> tensor<1x112x12x12xf16, {order = #NHCW}>

}
