//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-all" %s | FileCheck --check-prefix=CHECK-ALL %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-init" %s | FileCheck --check-prefix=CHECK-INIT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-main" %s | FileCheck --check-prefix=CHECK-MAIN %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// Note: these tests verify ws-extraction-mode differences of the
// introduce-init-function pass. They are not supposed to test everything but
// rather test the bare minimum, focusing on the difference in the mode.

{-#
    dialect_resources: {
        builtin: {
            ov_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @TestAllOptions {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
        DataInfo "output2" : tensor<4x16xf32>
    }

    func.func @main(%input: tensor<4x16xf16>) -> (tensor<2x2xf32>, tensor<4x16xf32>) {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<ov_1> : tensor<4x4xf32>,
            [#const.Add<1.0 : f32>, #const.SubView<[2, 2], [2, 2]>]
        %out = IE.Convert(%input) {dstElemType = f32} : tensor<4x16xf16> -> tensor<4x16xf32>
        return %cst, %out : tensor<2x2xf32>, tensor<4x16xf32>
    }
}

// CHECK-ALL-LABEL:     @TestAllOptions
// CHECK-ALL:           net.NetworkInfo entryPoint : @wrapper_main
// CHECK-ALL:               inputsInfo : {
// CHECK-ALL-NEXT:              DataInfo "input1" : tensor<4x16xf16>
// CHECK-ALL:               outputsInfo : {
// CHECK-ALL-NEXT:              DataInfo "output1" : tensor<2x2xf32>
// CHECK-ALL-NEXT:              DataInfo "output2" : tensor<4x16xf32>

// CHECK-ALL:           func.func private @init([[ORIG_CST:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-ALL-NEXT:          [[ADDEND:%.+]] = const.Declare tensor<1xf32>
// CHECK-ALL-NEXT:          [[RES:%.+]] = IE.Add([[ORIG_CST]], [[ADDEND]])
// CHECK-ALL-NEXT:          return [[RES]]

// CHECK-ALL:           func.func private @main([[IN:%.+]]: tensor<4x16xf16>, [[PREV_CST:%.+]]: tensor<4x4xf32>)
// CHECK-ALL-SAME:               -> (tensor<2x2xf32>, tensor<4x16xf32>)
// CHECK-ALL-NEXT:          [[SLICE:%.+]] = VPU.Slice [[PREV_CST]] [2, 2] [2, 2]
// CHECK-ALL-NEXT:          [[CVT:%.+]] = IE.Convert([[IN]]) {dstElemType = f32}
// CHECK-ALL-NEXT:          return [[SLICE]], [[CVT]]

// CHECK-ALL:           func.func @wrapper_main([[IN:%.+]]: tensor<4x16xf16>) -> (tensor<2x2xf32>, tensor<4x16xf32>)
// CHECK-ALL-NEXT:          [[CST:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<ov_1>
// CHECK-ALL-NEXT:          [[INIT_CST:%.+]] = call @init([[CST]])
// CHECK-ALL-NEXT:          [[MAIN_RES:%.+]]:2 = call @main([[IN]], [[INIT_CST]])
// CHECK-ALL-NEXT:          return [[MAIN_RES]]#0, [[MAIN_RES]]#1


// CHECK-INIT-LABEL:    @TestAllOptions
// CHECK-INIT:          net.NetworkInfo entryPoint : @init
// CHECK-INIT:              inputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "in_ov_1" : tensor<4x4xf32>
// CHECK-INIT:              outputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "out_ov_1_hash_11258667776708180655" : tensor<4x4xf32>

// CHECK-INIT:          func.func @init([[ORIG_CST:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-INIT-NEXT:         [[ADDEND:%.+]] = const.Declare tensor<1xf32>
// CHECK-INIT-NEXT:         [[RES:%.+]] = IE.Add([[ORIG_CST]], [[ADDEND]])
// CHECK-INIT-NEXT:         return [[RES]]

// CHECK-INIT-NOT:      func.func private @main
// CHECK-INIT-NOT:      func.func @wrapper_main


// CHECK-MAIN-LABEL:    @TestAllOptions
// CHECK-MAIN:          net.NetworkInfo entryPoint : @main
// CHECK-MAIN:              inputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "input1" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_1_hash_11258667776708180655" : tensor<4x4xf32>
// CHECK-MAIN:              outputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "output1" : tensor<2x2xf32>
// CHECK-MAIN-NEXT:             DataInfo "output2" : tensor<4x16xf32>

// CHECK-MAIN-NOT:      func.func private @init

// CHECK-MAIN:          func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[PREV_CST:%.+]]: tensor<4x4xf32>)
// CHECK-MAIN-SAME:              -> (tensor<2x2xf32>, tensor<4x16xf32>)
// CHECK-MAIN-NEXT:         [[SLICE:%.+]] = VPU.Slice [[PREV_CST]] [2, 2] [2, 2]
// CHECK-MAIN-NEXT:         [[CVT:%.+]] = IE.Convert([[IN]]) {dstElemType = f32}
// CHECK-MAIN-NEXT:         return [[SLICE]], [[CVT]]

// CHECK-MAIN-NOT:      func.func private @wrapper_main

// -----

// This test verifies that outlined functions are correctly handled in weights
// separation.

{-#
    dialect_resources: {
        builtin: {
            ov_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @OutlinedConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
        DataInfo "output2" : tensor<4x16xf16>
        DataInfo "output3" : tensor<4x4xf32>
    }

    func.func private @main_part1() -> tensor<4x4xf32> {
        %cst = const.Declare tensor<4x4xf32> = dense_resource<ov_1> : tensor<4x4xf32>, [#const.Add<5.0 : f32>]
        return %cst : tensor<4x4xf32>
    }

    func.func @main(%input: tensor<4x16xf16>) -> (tensor<2x2xf32>, tensor<4x16xf16>, tensor<4x4xf32>) {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<ov_1> : tensor<4x4xf32>,
            [#const.Add<1.0 : f32>, #const.SubView<[2, 2], [2, 2]>]
        // Note: called twice to catch additional bugs
        %out = call @main_part1() : () -> tensor<4x4xf32>
        %out2 = call @main_part1() : () -> tensor<4x4xf32>
        return %cst, %input, %out2 : tensor<2x2xf32>, tensor<4x16xf16>, tensor<4x4xf32>
    }

// CHECK-ALL-LABEL:     @OutlinedConstants
// CHECK-ALL:           net.NetworkInfo entryPoint : @wrapper_main
// CHECK-ALL:              inputsInfo : {
// CHECK-ALL-NEXT:             DataInfo "input1" : tensor<4x16xf16>
// CHECK-ALL:              outputsInfo : {
// CHECK-ALL-NEXT:             DataInfo "output1" : tensor<2x2xf32>
// CHECK-ALL-NEXT:             DataInfo "output2" : tensor<4x16xf16>
// CHECK-ALL-NEXT:             DataInfo "output3" : tensor<4x4xf32>

// CHECK-ALL:           func.func private @main_part1
// CHECK-ALL:           func.func private @init
// CHECK-ALL:           func.func private @main
// CHECK-ALL:           func.func @wrapper_main


// CHECK-INIT-LABEL:    @OutlinedConstants
// CHECK-INIT:          net.NetworkInfo entryPoint : @init
// CHECK-INIT:              inputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "in_ov_1" : tensor<4x4xf32>
// CHECK-INIT:              outputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "out_ov_1_hash_11258667776708180655" : tensor<4x4xf32>
// CHECK-INIT-NEXT:             DataInfo "out_ov_1_hash_4063002564071487318" : tensor<4x4xf32>

// CHECK-INIT-NOT:      func.func private @main_part1
// CHECK-INIT:          func.func @init
// CHECK-INIT-NOT:      func.func private @main
// CHECK-INIT-NOT:      func.func @wrapper_main


// CHECK-MAIN-LABEL:    @OutlinedConstants
// CHECK-MAIN:          net.NetworkInfo entryPoint : @main
// CHECK-MAIN:              inputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "input1" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_1_hash_11258667776708180655" : tensor<4x4xf32>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_1_hash_4063002564071487318" : tensor<4x4xf32>
// CHECK-MAIN:              outputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "output1" : tensor<2x2xf32>
// CHECK-MAIN-NEXT:             DataInfo "output2" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "output3" : tensor<4x4xf32>

// CHECK-MAIN:          func.func private @main_part1
// CHECK-MAIN-NOT:      func.func private @init
// CHECK-MAIN:          func.func @main
// CHECK-MAIN-NOT:      func.func private @wrapper_main
}

// -----

// This test verifies that constant hashes are computed correctly.

{-#
    dialect_resources: {
        builtin: {
            ov_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd",
            ov_42: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @HashConsistency {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x4xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x4xf16>
    }

    func.func @main(%input: tensor<4x4xf16>) -> tensor<4x4xf16> {
        // Note: {ov1, ov42}_common_hash share the exact same "in init"
        // transformations -> this means their hashes would match
        %ov1_common_hash = const.Declare tensor<1x1xf16> = dense_resource<ov_1> : tensor<4x4xf32>,
            [#const.CastElemType<f16>, #const.Rescale<3.0>, #const.SubView<[0, 0], [1, 1]>]
        %ov42_common_hash = const.Declare tensor<1x1xf16> = dense_resource<ov_42> : tensor<2x8xf32>,
            [#const.CastElemType<f16>, #const.Rescale<3.0>, #const.SubView<[2, 2], [1, 1]>]

        %ov1_unique_hash = const.Declare tensor<1x1xf16> = dense_resource<ov_1> : tensor<4x4xf32>,
            [#const.CastElemType<f16>, #const.Add<42.0>, #const.SubView<[0, 0], [1, 1]>]

        %0 = VPU.Add(%input, %ov1_common_hash) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<4x4xf16>, tensor<1x1xf16> -> tensor<4x4xf16>
        %1 = VPU.Add(%0, %ov42_common_hash) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<4x4xf16>, tensor<1x1xf16> -> tensor<4x4xf16>
        %2 = VPU.Add(%1, %ov1_unique_hash) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<4x4xf16>, tensor<1x1xf16> -> tensor<4x4xf16>

        return %2 : tensor<4x4xf16>
    }

// Note: CHECK-ALL also checks how IR generation looks to verify the basics

// CHECK-ALL-LABEL:     @HashConsistency
// CHECK-ALL:   func.func private @init([[OV_1:%.+]]: tensor<4x4xf32>, [[OV_42:%.+]]: tensor<2x8xf32>)
// CHECK-ALL-SAME:  -> (tensor<4x4xf16>, tensor<4x4xf16>, tensor<2x8xf16>)
// CHECK-ALL:       [[CAST_1:%.+]] = IE.Convert([[OV_1]]) {dstElemType = f16}

// CHECK-ALL:       [[FOURTYTWO:%.+]] = const.Declare tensor<1xf16> = dense<4.200000e+01>
// CHECK-ALL:       [[ADD_1:%.+]] = IE.Add([[CAST_1]], [[FOURTYTWO]])

// CHECK-ALL:       [[THREE0:%.+]] = const.Declare tensor<1xf16> = dense<3.000000e+00>
// CHECK-ALL:       [[RESCALE_1:%.+]] = IE.Multiply([[CAST_1]], [[THREE0]])

// CHECK-ALL:       [[CAST_42:%.+]] = IE.Convert([[OV_42]]) {dstElemType = f16}
// CHECK-ALL:       [[THREE1:%.+]] = const.Declare tensor<1xf16> = dense<3.000000e+00>
// CHECK-ALL:       [[RESCALE_42:%.+]] = IE.Multiply([[CAST_42]], [[THREE1]])

// CHECK-ALL:       return [[ADD_1]], [[RESCALE_1]], [[RESCALE_42]]


// CHECK-ALL:   func.func private @main([[INPUT:%.+]]: tensor<4x4xf16>, [[ADD_1:%.+]]: tensor<4x4xf16>, [[RESCALE_1:%.+]]: tensor<4x4xf16>, [[RESCALE_42:%.+]]: tensor<2x8xf16>)
// CHECK-ALL-SAME:  -> tensor<4x4xf16>
// CHECK-ALL:       [[SUBVIEW_ADD_1:%.+]] = VPU.Slice [[ADD_1]] [0, 0] [1, 1]
// CHECK-ALL:       [[SUBVIEW_RESCALE_1:%.+]] = VPU.Slice [[RESCALE_1]] [0, 0] [1, 1]
// CHECK-ALL:       [[SUBVIEW_RESCALE_42:%.+]] = VPU.Slice [[RESCALE_42]] [2, 2] [1, 1]
// CHECK-ALL:       [[CHAIN0:%.+]] = VPU.Add([[INPUT]], [[SUBVIEW_RESCALE_1]])
// CHECK-ALL:       [[CHAIN1:%.+]] = VPU.Add([[CHAIN0]], [[SUBVIEW_RESCALE_42]])
// CHECK-ALL:       [[CHAIN2:%.+]] = VPU.Add([[CHAIN1]], [[SUBVIEW_ADD_1]])
// CHECK-ALL:       return [[CHAIN2]]


// CHECK-ALL:   func.func @wrapper_main([[INPUT:%.+]]: tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK-ALL:       [[OV_1:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<ov_1>
// CHECK-ALL:       [[OV_42:%.+]] = const.Declare tensor<2x8xf32> = dense_resource<ov_42>
// CHECK-ALL:       [[INIT:%.+]]:3 = call @init([[OV_1]], [[OV_42]])
// CHECK-ALL:       [[MAIN:%.+]] = call @main([[INPUT]], [[INIT]]#0, [[INIT]]#1, [[INIT]]#2)
// CHECK-ALL:       return [[MAIN]]


// CHECK-INIT-LABEL:    @HashConsistency
// CHECK-INIT:          net.NetworkInfo entryPoint : @init
// CHECK-INIT:              inputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "in_ov_42" : tensor<2x8xf32>
// CHECK-INIT-NEXT:             DataInfo "in_ov_1" : tensor<4x4xf32>
// CHECK-INIT:              outputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "out_ov_42_hash_6705143075530545067" : tensor<2x8xf16>
// CHECK-INIT-NEXT:             DataInfo "out_ov_1_hash_7071254137056153727" : tensor<4x4xf16>
// CHECK-INIT-NEXT:             DataInfo "out_ov_1_hash_6705143075530545067" : tensor<4x4xf16>

// CHECK-INIT:  func.func @init([[OV_42:%.+]]: tensor<2x8xf32>, [[OV_1:%.+]]: tensor<4x4xf32>)
// CHECK-INIT-SAME: -> (tensor<2x8xf16>, tensor<4x4xf16>, tensor<4x4xf16>)


// CHECK-MAIN-LABEL:    @HashConsistency
// CHECK-MAIN:          net.NetworkInfo entryPoint : @main
// CHECK-MAIN:              inputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "input1" : tensor<4x4xf16>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_1_hash_7071254137056153727" : tensor<4x4xf16>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_1_hash_6705143075530545067" : tensor<4x4xf16>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_42_hash_6705143075530545067" : tensor<2x8xf16>
// CHECK-MAIN:              outputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "output1" : tensor<4x4xf16>

// CHECK-MAIN:  func.func @main([[INPUT:%.+]]: tensor<4x4xf16>, [[OV_1_ADD:%.+]]: tensor<4x4xf16>, [[OV_1_RESCALE:%.+]]: tensor<4x4xf16>, [[OV_42_RESCALE:%.+]]: tensor<2x8xf16>)
// CHECK-MAIN-SAME: -> tensor<4x4xf16>
}
