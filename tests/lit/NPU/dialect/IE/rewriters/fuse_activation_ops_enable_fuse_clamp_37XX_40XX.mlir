//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-activation-ops="enable-fuse-clamp=true" %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --run-adjust-for-vpu-rewriters="enable-fuse-clamp=true rewriter=fuse-activation-ops-set" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @NotFuseMaxPoolWithClampTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @NotFuseMaxPoolWithClampTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Clamp(%0)
        {
            max = 6.000000e+00 : f64,
            min = 0.000000e+00 : f64
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]])
    // CHECK: [[CLAMP:%.+]] = IE.Clamp([[MAX_POOL]])

    // CHECK: return [[CLAMP]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 12.695739985447304:118>

// CHECK-LABEL: func.func @NotFuseQuantMaxPoolWithClampTest
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x32x97x177x!qElemType>
func.func @NotFuseQuantMaxPoolWithClampTest(%arg0: tensor<1x32x97x177x!qElemType>) -> tensor<1x32x96x176x!qElemType> {
    %0 = IE.MaxPool(%arg0)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } :
        tensor<1x32x97x177x!qElemType> -> tensor<1x32x96x176x!qElemType>

    %1 = IE.Clamp(%0)
        {
            max = 6.000000e+00 : f64,
            min = 0.000000e+00 : f64
        } :
        tensor<1x32x96x176x!qElemType> -> tensor<1x32x96x176x!qElemType>

    return %1 : tensor<1x32x96x176x!qElemType>

    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]])
    // CHECK: [[CLAMP:%.+]] = IE.Clamp([[MAX_POOL]])

    // CHECK: return [[CLAMP]] : tensor<1x32x96x176x!qElemType>
}
