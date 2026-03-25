//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --default-hw-mode-ie %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: @NoFuseReduceSquare
module @NoFuseReduceSquare {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x32x32x96xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x32x1xf16>
    }

    func.func @main(%arg0: tensor<1x32x32x96xf32>) -> tensor<1x32x32x1xf32> {
        %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
        %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf32>
        %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
        %2 = IE.Sqrt(%1) : tensor<1x32x32x1xf32> -> tensor<1x32x32x1xf32>
        return %2 : tensor<1x32x32x1xf32>

        // CHECK-NOT: IE.ReduceSquare
    }
}

// -----

// CHECK-LABEL: @NoFuseReduceSquareWithAdd
module @NoFuseReduceSquareWithAdd {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x32x32x96xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x32x1xf16>
    }

    func.func @main(%arg0: tensor<1x32x32x96xf16>) -> tensor<1x32x32x1xf16> {
        %cst = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
        %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
        %0 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
        %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
        %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x1xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x1xf16>
        %3 = IE.Sqrt(%2) : tensor<1x32x32x1xf16> -> tensor<1x32x32x1xf16>
        return %3 : tensor<1x32x32x1xf16>

        // CHECK-NOT: IE.ReduceSquare
    }
}
