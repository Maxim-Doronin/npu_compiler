//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --host-compile %s | FileCheck %s --check-prefixes=CHECK,CHECK-%arch%

// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @EltwiseNHWCDynamic
module @EltwiseNHWCDynamic {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x?x1000xf16>
        DataInfo "input2" : tensor<1x16x?x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x?x1000xf16>
    }

    func.func @main(%arg0: tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}> {
      %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>,
            tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>
                -> tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>
      return %0 : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }

    // CHECK: func.func @output_shape([[ARG0:%.+]]: memref<1x16x?x1000xf16, #NHWC>, [[ARG1:%.+]]: memref<1x16x?x1000xf16, #NHWC>, [[ARG2:%.+]]: memref<4xi64>) -> memref<4xi64> attributes {[[ANY_ATTR:.+]]} {
    // CHECK:    [[CST_3:%.+]] = arith.constant 3 : index
    // CHECK:    [[CST_1:%.+]] = arith.constant 1 : index
    // CHECK:    [[CST_0:%.+]] = arith.constant 0 : index
    // CHECK:    [[CST_1000_i64:%.+]] = arith.constant 1000 : i64
    // CHECK:    [[CST_2:%.+]] = arith.constant 2 : index
    // CHECK:    [[CST_16_i64:%.+]] = arith.constant 16 : i64
    // CHECK:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
    // CHECK:    [[DIM:%.+]] = memref.dim [[ARG0]], [[CST_2]] : memref<1x16x?x1000xf16, #NHWC>
    // CHECK:    [[IDX_CAST:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK:    memref.store [[CST_1_i64]], [[ARG2]][[[CST_0]]] : memref<4xi64>
    // CHECK:    memref.store [[CST_16_i64]], [[ARG2]][[[CST_1]]] : memref<4xi64>
    // CHECK:    memref.store [[IDX_CAST]], [[ARG2]][[[CST_2]]] : memref<4xi64>
    // CHECK:    memref.store [[CST_1000_i64]], [[ARG2]][[[CST_3]]] : memref<4xi64>
    // CHECK:    return [[ARG2]] : memref<4xi64>
    // }

    // CHECK: func.func @main_func0_static([[func0_ARG0:%.+]]: memref<1x[[STEP:.+]]x1000x16xf16>, [[func0_ARG1:%.+]]: memref<1x[[STEP]]x1000x16xf16>, [[func0_ARG2:%.+]]: memref<1x[[STEP]]x1000x16xf16>) -> memref<1x[[STEP]]x1000x16xf16> {

    // CHECK-NPU40XX-COUNT-6: VPUIP.NCEClusterTask
    // CHECK-NPU50XX-COUNT-3: VPUIP.NCEClusterTask
    // CHECK-NOT: IE.Add

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x?x1000x16xf16>, [[ARG1:%.+]]: memref<1x?x1000x16xf16>, [[ARG2:%.+]]: memref<1x?x1000x16xf16>) -> memref<1x?x1000x16xf16> attributes {[[ANY_ATTR:.+]]} {
    // CHECK: [[STEP_VAR:%.+]] = arith.constant [[STEP]] : index
    // CHECK: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[C1:%.+]] = arith.constant 1 : index
    // CHECK: [[DIM:%.+]] = memref.dim [[ARG0]], [[C1]] : memref<1x?x1000x16xf16>
    // CHECK: [[CMP_SGE_OUTER:%.+]] = arith.cmpi sge, [[DIM]], [[STEP_VAR]] : index
    // CHECK: cf.assert [[CMP_SGE_OUTER]], "Not enough elements to backtrack in scf.for loop for Output tensor"
    // CHECK: [[SUB:%.+]] = arith.subi [[DIM]], [[C0]] : index
    // CHECK: [[DIV:%.+]] = arith.divsi [[SUB]], [[STEP_VAR]] : index
    // CHECK: [[GROUP:%.+]] = async.create_group [[DIV]] : !async.group
    // CHECK: scf.for [[ARG3:%.+]] = [[C0]] to [[DIM]] step [[STEP_VAR]] {
    // CHECK:   [[OFFSET:%.+]] = affine.min #map([[ARG3]]){{\[}}[[DIM]]{{\]}}
    // CHECK:   [[SUBVIEW0:%.+]] = memref.subview [[ARG0]][0, [[OFFSET]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK:   [[SUBVIEW1:%.+]] = memref.subview [[ARG1]][0, [[OFFSET]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK:   [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW0]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK:   [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW1]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK:   [[SUBVIEW2:%.+]] = memref.subview [[ARG2]][0, [[OFFSET]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK:   [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW2]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK:   [[TOKEN:%.+]], [[BODY_RESULTS:%.+]] = async.execute -> !async.value<memref<1x[[STEP]]x1000x16xf16>> {
    // CHECK:     [[NESTED_CALL:%.+]] = Core.NestedCall @Module0::@main_func0_static([[CAST0]], [[CAST1]], [[CAST2]]) : (memref<1x[[STEP]]x1000x16xf16>, memref<1x[[STEP]]x1000x16xf16>, memref<1x[[STEP]]x1000x16xf16>) -> memref<1x[[STEP]]x1000x16xf16>
    // CHECK:     async.yield [[NESTED_CALL]] : memref<1x[[STEP]]x1000x16xf16>
    // CHECK:   }
    // CHECK:   [[ADD_TO_GROUP:%.+]] = async.add_to_group [[TOKEN]], [[GROUP]] : !async.token
    // CHECK:   [[AWAIT:%.+]] = async.await [[BODY_RESULTS]] : !async.value<memref<1x[[STEP]]x1000x16xf16>>
    // CHECK: }
    // CHECK: async.await_all [[GROUP]]
    // CHECK: return [[ARG2]] : memref<1x?x1000x16xf16>
}
