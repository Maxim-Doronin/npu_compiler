//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --ws-monolithic-partial %s | FileCheck %s --strict-whitespace
// TODO: #-157476 Enable LIT test pipeline for other architectures
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @WeightsSeprationMode
{-#
    dialect_resources: {
        builtin: {
            ov_1: "0x0000000400aa"
        }
    }
#-}

module @WeightsSeprationMode attributes {} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x2x1x1xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2x1x1xui8>
    }

    func.func @main(%arg0: tensor<1x2x1x1xui8>) -> tensor<1x2x1x1xui8> {
        %cst = const.Declare tensor<1x2x1x1xui8> = dense_resource<ov_1> : tensor<1x2x1x1xui8>, [#const.Add<1.0 : f32>]
        return %cst : tensor<1x2x1x1xui8>
    }

// Note: We mainly want to check that #const.Add is mapped to a VPU.Add and don't care about any of the other functionality
//       the pipelines perform.
// CHECK:  func.func @wrapper_main([[ARG0:%.+]]: tensor<1x2x1x1xui8>) -> tensor<1x2x1x1xui8> {
// CHECK:      [[CST:%.+]] = const.Declare tensor<1x2x1x1xui8> = dense_resource<ov_1> : tensor<1x2x1x1xui8>
// CHECK:      [[ADD:%.+]] = VPU.Add
// CHECK:      return
}
