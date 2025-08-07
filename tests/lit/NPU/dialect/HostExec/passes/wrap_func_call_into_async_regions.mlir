//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wrap-func-calls-into-async-regions %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (240, -d0 + s0)>
module @ControlFlowOutliningDynamicShape  {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x256x?xf16>
        DataInfo "input2" : tensor<1x16x256x?xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x16x256x?xf16>
    }
    module @Module0 {
        func.func private @main_func0(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                                      %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
                          -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
            %0 = VPU.NCE.Eltwise(%arg0, %arg1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                    clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00],
                    fp_prelu_alpha = 1.000000e+00 : f64>}
                -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
            return %0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        }
    }

    func.func @main(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                    %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
               -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
        %c3 = arith.constant 3 : index
        %dim = tensor.dim %arg0, %c3 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        %0 = tensor.empty(%dim) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        %c0 = arith.constant 0 : index
        %c240 = arith.constant 240 : index
        %1 = scf.for %arg2 = %c0 to %dim step %c240 iter_args(%arg3 = %0) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {
            %2 = affine.min #map(%arg2)[%dim]
            %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1]
                             : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
                             to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
            %extracted_slice_2 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1]
                               : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
                               to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

            %3 = Core.NestedCall @Module0::@main_func0(%extracted_slice, %extracted_slice_2)
               : (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                  tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
               -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

            %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1]
                            : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
                            into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
            scf.yield %inserted_slice : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        }

        %2 = Core.NestedCall @Module0::@main_func0(%1, %arg1)
               : (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                  tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
               -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

        return %2 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    }
    // CHECK-LABEL: @ControlFlowOutliningDynamicShape

    // CHECK: module [[MODULE0:@.+]] {
    // CHECK: func.func private [[FUNC0:@.+]](%arg0: tensor<1x16x256x?xf16

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME: [[ARG1:%.+]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
    // CHECK: [[C3:%.+]] = arith.constant 3 : index
    // CHECK: [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]]
    // CHECK: [[ALLOC:%.+]] = tensor.empty([[DIM]])
    // CHECK: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[C240:%.+]] = arith.constant 240 : index
    // CHECK: [[SUB:%.+]] = arith.subi [[DIM]], [[C0]]
    // CHECK: [[DIV:%.+]] = arith.divsi [[SUB]], [[C240]]
    // CHECK: [[GROUP:%.+]] = async.create_group [[DIV]]

    // CHECK: [[FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[DIM]] step [[C240]] iter_args([[ARG3:%.+]] = [[ALLOC]])
    // CHECK: [[TOKEN:%.+]], [[RESULTS:%.+]] = async.execute
    // CHECK: Core.NestedCall [[MODULE0]]::[[FUNC0]]

    // CHECK: async.add_to_group [[TOKEN]], [[GROUP]] : !async.token
    // CHECK: async.await [[RESULTS]]

    // CHECK: scf.yield

    // CHECK: async.await_all [[GROUP]]

    // CHECK: [[TOKEN0:%.+]], [[RESULTS0:%.+]] = async.execute
    // CHECK: Core.NestedCall [[MODULE0]]::[[FUNC0]]
    // CHECK: [[RETURN:%.+]] = async.await [[RESULTS0]]

    // CHECK: return [[RETURN]]
}
