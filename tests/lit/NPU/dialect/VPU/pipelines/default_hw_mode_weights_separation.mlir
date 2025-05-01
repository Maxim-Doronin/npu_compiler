//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpu="ws-extraction-mode=gen-all" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WeightsSeprationMode
module @WeightsSeprationMode attributes {} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1x1x1xsi8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x1x1xsi8>
    }

    func.func @main(%arg0: tensor<1x1x1x1xsi8>) -> tensor<1x1x1x1xsi8> {
        %cst = const.Declare tensor<1x1x1x1xsi8> = dense<1> : tensor<1x1x1x1xsi8>
        return %cst : tensor<1x1x1x1xsi8>
    }

// CHECK:  func.func private @init

// CHECK:  func.func private @main

// CHECK:  func.func @wrapper_main

}
