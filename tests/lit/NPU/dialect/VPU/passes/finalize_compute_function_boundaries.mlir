//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --finalize-compute-function-boundaries  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @StaticEltwiseNHWC {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x720x1000xf16>
        DataInfo "input2" : tensor<1x16x720x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x720x1000xf16>
    }

    func.func private @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>) -> tensor<1x16x90x1000xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        return %0 : tensor<1x16x90x1000xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x16x720x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16, {order = #NHWC}> {
        %c90 = arith.constant 90 : index
        %c720 = arith.constant 720 : index
        %c0 = arith.constant 0 : index
        %0 = tensor.empty() : tensor<1x16x720x1000xf16, {order = #NHWC}>
        %1 = scf.for %arg2 = %c0 to %c720 step %c90 iter_args(%arg3 = %0) -> (tensor<1x16x720x1000xf16, {order = #NHWC}>) {
            %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x720x1000xf16, {order = #NHWC}> to tensor<1x16x90x1000xf16, {order = #NHWC}>
            %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x720x1000xf16, {order = #NHWC}> to tensor<1x16x90x1000xf16, {order = #NHWC}>
            %2 = func.call @main_func0(%extracted_slice, %extracted_slice_0) : (tensor<1x16x90x1000xf16, {order = #NHWC}>, tensor<1x16x90x1000xf16, {order = #NHWC}>) -> tensor<1x16x90x1000xf16, {order = #NHWC}>
            %inserted_slice = tensor.insert_slice %2 into %arg3[0, 0, %arg2, 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x90x1000xf16, {order = #NHWC}> into tensor<1x16x720x1000xf16, {order = #NHWC}>
            scf.yield %inserted_slice : tensor<1x16x720x1000xf16, {order = #NHWC}>
        }
        return %1 : tensor<1x16x720x1000xf16, {order = #NHWC}>
    }
}

// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @StaticEltwiseNHWC
// CHECK: func.func private @main_func0([[ARG0:%.+]]: tensor<1x90x1000x16xf16>, [[ARG1:%.+]]: tensor<1x90x1000x16xf16>)
// CHECK-DAG: [[CAST0:%.+]] = VPU.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW}
// CHECK-DAG: [[CAST1:%.+]] = VPU.PermuteCast([[ARG1]]) {dst_order = #NHWC, mem_perm = #NCHW}
// CHECK:     [[ADD:%.+]] = VPU.NCE.Eltwise([[CAST0]], [[CAST1]])
// CHECK:     [[CAST2:%.+]] = VPU.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW}
// CHECK:     return [[CAST2]] : tensor<1x90x1000x16xf16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x720x1000x16xf16>, [[ARG1:%.+]]: tensor<1x720x1000x16xf16>)
// CHECK:     [[C90:%.+]] = arith.constant 90 : index
// CHECK:     [[C720:%.+]] = arith.constant 720 : index
// CHECK:     [[C0:%.+]] = arith.constant 0 : index
// CHECK:     [[ALLOC:%.+]] = tensor.empty() : tensor<1x720x1000x16xf16>
// CHECK:     [[FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C720]] step [[C90]] iter_args([[ARG3:%.+]])
// CHECK:         [[EXTRACTED_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, [[ARG2]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:         [[EXTRACTED_SLICE_0:%.+]] = tensor.extract_slice [[ARG1]][0, [[ARG2]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:         [[CALL:%.+]] = func.call @main_func0([[EXTRACTED_SLICE]], [[EXTRACTED_SLICE_0]])
// CHECK:         [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[CALL]]
// CHECK:         scf.yield [[INSERTED_SLICE]]
// CHECK:     return [[FOR]] : tensor<1x720x1000x16xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @StaticEltwiseMultipleOps {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x90x1000xf16>
        DataInfo "input2" : tensor<1x16x90x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x90x1000xf16>
    }

    func.func private @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>)
        -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        %1 = VPU.NCE.Eltwise(%0, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NHWC}>
        %2 = VPU.NCE.Eltwise(%1, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
            -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %2 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
    func.func @main(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>)
              -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = func.call @main_func0(%arg0, %arg1)
           : (tensor<1x16x90x1000xf16, {order = #NHWC}>, tensor<1x16x90x1000xf16, {order = #NHWC}>)
           -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
}

// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


// CHECK-LABEL: @StaticEltwiseMultipleOps
// CHECK: func.func private @main_func0([[ARG0:%.+]]: tensor<1x90x1000x16xf16>, [[ARG1:%.+]]: tensor<1x90x1000x16xf16>)
// CHECK-DAG: [[CAST0:%.+]] = VPU.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW}
// CHECK-DAG: [[CAST1:%.+]] = VPU.PermuteCast([[ARG1]]) {dst_order = #NHWC, mem_perm = #NCHW}
// CHECK:     [[ADD0:%.+]] = VPU.NCE.Eltwise([[CAST0]], [[CAST1]])
// CHECK:     [[ADD1:%.+]] = VPU.NCE.Eltwise([[ADD0]], [[CAST1]])
// CHECK:     [[ADD2:%.+]] = VPU.NCE.Eltwise([[ADD1]], [[CAST1]])
// CHECK:     [[CAST2:%.+]] = VPU.PermuteCast([[ADD2]]) {dst_order = #NCHW, mem_perm = #NCHW}
// CHECK:     return [[CAST2]] : tensor<1x16x90x1000xf16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x90x1000x16xf16>, [[ARG1:%.+]]: tensor<1x90x1000x16xf16>)
// CHECK:    [[CALL:%.+]] = call @main_func0([[ARG0]], [[ARG1]])
// CHECK:    return [[CALL]] : tensor<1x16x90x1000xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @StaticEltwiseNCHW {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x90x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x90x1000xf16>
    }

    func.func private @main_func0(%arg0: tensor<1x16x90x1000xf16, {order = #NCHW}>) -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = VPU.ReLU(%arg0) : tensor<1x16x90x1000xf16, {order = #NCHW}> -> tensor<1x16x90x1000xf16, {order = #NCHW}>
        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }

    func.func @main(%arg0: tensor<1x16x90x1000xf16, {order = #NCHW}>) -> tensor<1x16x90x1000xf16, {order = #NCHW}> {
        %0 = func.call @main_func0(%arg0)
           : (tensor<1x16x90x1000xf16, {order = #NCHW}>)
           -> tensor<1x16x90x1000xf16, {order = #NCHW}>

        return %0 : tensor<1x16x90x1000xf16, {order = #NCHW}>
    }
}

// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @StaticEltwiseNCHW
// CHECK: func.func private @main_func0([[ARG0:%.+]]: tensor<1x16x90x1000xf16>)
// CHECK:     [[CAST0:%.+]] = Core.ReinterpretCast([[ARG0]]) : tensor<1x16x90x1000xf16> -> tensor<1x16x90x1000xf16, {order = #NCHW}>
// CHECK:     [[RELU:%.+]] = VPU.ReLU([[CAST0]])
// CHECK:     [[CAST1:%.+]] = VPU.PermuteCast([[RELU]]) {dst_order = #NCHW, mem_perm = #NCHW}
// CHECK:     return [[CAST1]] : tensor<1x16x90x1000xf16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x16x90x1000xf16>)
// CHECK:    [[CALL:%.+]] = call @main_func0([[ARG0]])
// CHECK:    return [[CALL]] : tensor<1x16x90x1000xf16>
