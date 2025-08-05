//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-all" --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


{-#
  dialect_resources: {
    builtin: {
      ov: "0x10000000ABABABABCDCDCDCD"
    }
  }
#-}

!qElemType = !quant.uniform<i8:f16, 0.5:128>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16, 5.000000e-01:128>

// CHECK: module @SimilarConstantsSimilarLocationNames
module @SimilarConstantsSimilarLocationNames {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Parameter_1" : tensor<2x2x1x1xf16>
    } outputsInfo : {
        DataInfo "Cst_1" friendlyName = "Result_1" : tensor<2x2x1x1xi8>
        DataInfo "Cst_2" friendlyName = "Result_2" : tensor<2x2x1x1xi8>
    }

    func.func @main(%arg0: tensor<2x2x1x1xf16>) -> (tensor<2x2x1x1xi8>, tensor<2x2x1x1xi8>) {
        %cst = const.Declare tensor<2x2x1x1x!qElemType> = dense_resource<ov> : tensor<2x2x1x1xf16>, [#const.SubView<[1, 0, 0, 0], [2, 2, 1, 1]>, #const.CastElemType<!qElemType>]
        %cst1 = const.Declare tensor<2x2x1x1x!qElemType> = dense_resource<ov> : tensor<2x2x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [2, 2, 1, 1]>, #const.CastElemType<!qElemType>]

        %0 = VPU.QuantizeCast(%cst) { dstElemType = i8 }
            : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xi8>
        %1 = VPU.QuantizeCast(%cst1) { dstElemType = i8 }
            : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xi8>
        return %0, %1: tensor<2x2x1x1xi8> , tensor<2x2x1x1xi8>
    }

    // CHECK-DAG: [[LOC_INIT0:#.+]] = loc("init_cstIdx_0")
    // CHECK-DAG: [[LOC_INIT1:#.+]] = loc("init_cstIdx_1")

    // CHECK-DAG: [[LOC_SLICE0:#.+]] = loc(fused[[[PRE0:#.+]], [[LOC_INIT0]], [[POST0:#.+]]])
    // CHECK-DAG: [[LOC_SLICE1:#.+]] = loc(fused[[[PRE1:#.+]], [[LOC_INIT1]], [[POST1:#.+]]])

    // CHECK-DAG: func.func private @init([[ARG0:%.+]]: tensor<2x2x1x1xf16> [[LOC_ARG0:.*]] -> (tensor<2x2x1x1xsi8>, tensor<2x2x1x1xsi8>)
    // CHECK-DAG: [[SLICEOP0:%.+]] = IE.Slice [[ARG0]] [1, 0, 0, 0] [2, 2, 1, 1] : tensor<2x2x1x1xf16> to tensor<2x2x1x1xf16> loc([[LOC_SLICE0]])
    // CHECK-DAG: [[CONVERT0:%.+]] = IE.Convert([[SLICEOP0]]) {dstElemType = i8} : tensor<2x2x1x1xf16> -> tensor<2x2x1x1xi8>
    // CHECK-DAG: [[QC0:%.+]] = IE.QuantizeCast([[CONVERT0]]) {dstElemType = [[QTYPE]]} : tensor<2x2x1x1xi8> -> tensor<2x2x1x1x!qElemType>
    // CHECK-DAG: [[QC1:%.+]] = IE.QuantizeCast([[QC0]]) {dstElemType = si8} : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xsi8>

    // CHECK-DAG: [[SLICEOP1:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [2, 2, 1, 1] : tensor<2x2x1x1xf16> to tensor<2x2x1x1xf16> loc([[LOC_SLICE1]])
    // CHECK-DAG: [[CONVERT1:%.+]] = IE.Convert([[SLICEOP1]]) {dstElemType = i8} : tensor<2x2x1x1xf16> -> tensor<2x2x1x1xi8>
    // CHECK-DAG: [[QC2:%.+]] = IE.QuantizeCast([[CONVERT1]]) {dstElemType = [[QTYPE]]} : tensor<2x2x1x1xi8> -> tensor<2x2x1x1x!qElemType>
    // CHECK-DAG: [[QC3:%.+]] = IE.QuantizeCast([[QC2]]) {dstElemType = si8} : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xsi8>
    // CHECK-DAG: return [[QC1]], [[QC3]]

    // CHECK-DAG: [[LOC_MAIN0:#.+]] = loc("main_cstIdx_0")
    // CHECK-DAG: [[LOC_MAIN1:#.+]] = loc("main_cstIdx_1")

    // CHECK-DAG: [[LOC_QC0:#.+]] = loc(fused[[[PRE2:#.+]], [[LOC_MAIN0]], [[POST2:#.+]]])
    // CHECK-DAG: [[LOC_QC1:#.+]] = loc(fused[[[PRE3:#.+]], [[LOC_MAIN1]], [[POST3:#.+]]])

    // CHECK-DAG: func.func private @main([[MAIN_ARG0:%.+]]: tensor<2x2x1x1xf16> [[LOC_MAIN_ARG0:.*]], [[MAIN_ARG1:%.+]]: tensor<2x2x1x1xsi8> [[LOC_MAIN_ARG1:.*]], [[MAIN_ARG2:%.+]]: tensor<2x2x1x1xsi8> [[LOC_MAIN_ARG2:.*]]) -> (tensor<2x2x1x1xi8>, tensor<2x2x1x1xi8>)
    // CHECK-DAG: [[MAIN_QC0:%.+]] = VPU.QuantizeCast([[MAIN_ARG1]]) {dstElemType = [[QTYPE]]} : tensor<2x2x1x1xsi8> -> tensor<2x2x1x1x!qElemType> loc([[LOC_QC0]])
    // CHECK-DAG: [[MAIN_QC1:%.+]] = VPU.QuantizeCast([[MAIN_ARG2]]) {dstElemType = [[QTYPE]]} : tensor<2x2x1x1xsi8> -> tensor<2x2x1x1x!qElemType> loc([[LOC_QC1]])
    // CHECK-DAG: [[MAIN_QC2:%.+]] = VPU.QuantizeCast([[MAIN_QC0]]) {dstElemType = i8} : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xi8>
    // CHECK-DAG: [[MAIN_QC3:%.+]] = VPU.QuantizeCast([[MAIN_QC1]]) {dstElemType = i8} : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xi8>
    // CHECK-DAG: return [[MAIN_QC2]], [[MAIN_QC3]]
}
