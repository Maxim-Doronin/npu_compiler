//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-all" %s | FileCheck --check-prefix=CHECK-ALL %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-main" %s | FileCheck --check-prefix=CHECK-MAIN %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

{-#
    dialect_resources: {
        builtin: {
            some_other_origin: "0x10000000AABBCCDD"
        }
    }
#-}

module @NoConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        // Note: name doesn't start with "ov" and thus constant is ignored
        %cst1 = const.Declare tensor<4xui8> = dense_resource<some_other_origin> : tensor<4xui8>, [#const.Add<1.0>]
        return %arg : tensor<4x16xf16>
    }

    // CHECK-ALL: func.func private @main([[IN:%.+]]: tensor<4x16xf16>) -> tensor<4x16xf16>
    // CHECK-ALL-NEXT:  {{%.+}} = {{.*}} dense_resource<some_other_origin> {{.*}} [#const.Add<1.000000e+00 : f64>]
    // CHECK-ALL-NEXT:  return [[IN]]

    // CHECK-ALL: func.func @wrapper_main([[IN:%.+]]: tensor<4x16xf16>) -> tensor<4x16xf16>
    // CHECK-ALL-NEXT:  call @init() : () -> ()
    // CHECK-ALL-NEXT:  [[MAIN:%.+]] = call @main([[IN]])
    // CHECK-ALL-NEXT:  return [[MAIN]]

    // CHECK-MAIN: func.func @main({{%.+}}: tensor<4x16xf16>) -> tensor<4x16xf16>
    // CHECK-MAIN-NEXT:  {{%.+}} = {{.*}} dense_resource<some_other_origin> {{.*}} [#const.Add<1.000000e+00 : f64>]
}
