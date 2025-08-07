//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: func.func @Roundtrip([[ARG0:%.+]]: tensor<1x1xf16>) -> tensor<1x1xf16>
func.func @Roundtrip(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    %0 = Core.ReinterpretCast(%arg0) : tensor<1x1xf16> -> tensor<2x1xi8>
    %1 = Core.ReinterpretCast(%0) : tensor<2x1xi8> -> tensor<1x1xf16>
    return %1 : tensor<1x1xf16>

    // CHECK: [[VAR0:%.+]] = Core.ReinterpretCast([[ARG0]]) : tensor<1x1xf16> -> tensor<2x1xi8>
    // CHECK: [[VAR1:%.+]] = Core.ReinterpretCast([[VAR0]]) : tensor<2x1xi8> -> tensor<1x1xf16>
    // CHECK: return [[VAR1]]
}

// -----

// Note: non-required information may optionally be kept or lost. it is not
// clear yet what the semantics should be for tensor encoding and memref layout

#CN = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: func.func @TensorEncoding([[ARG0:%.+]]: tensor<1x1xf16, {order = #CN}>)
// CHECK-SAME: -> (tensor<1x1xf16>, tensor<2x1xi8, {order = #CN}>)
func.func @TensorEncoding(%arg0: tensor<1x1xf16, {order = #CN}>)
        -> (tensor<1x1xf16>, tensor<2x1xi8, {order = #CN}>) {
    %0 = Core.ReinterpretCast(%arg0) : tensor<1x1xf16, {order = #CN}> -> tensor<1x1xf16>
    %1 = Core.ReinterpretCast(%arg0) : tensor<1x1xf16, {order = #CN}> -> tensor<2x1xi8, {order = #CN}>
    return %0, %1 : tensor<1x1xf16>, tensor<2x1xi8, {order = #CN}>

    // CHECK: [[VAR0:%.+]] = Core.ReinterpretCast([[ARG0]])
    // CHECK-SAME: -> tensor<1x1xf16>
    // CHECK: [[VAR1:%.+]] = Core.ReinterpretCast([[ARG0]])
    // CHECK-SAME: -> tensor<2x1xi8, {order = #CN}>
    // CHECK: return [[VAR0]], [[VAR1]]
}

// -----

// CHECK: func.func @RankChange([[ARG0:%.+]]: tensor<1x1xf16>)
// CHECK-SAME: -> tensor<1xf16>
func.func @RankChange(%arg0: tensor<1x1xf16>) -> tensor<1xf16> {
    %0 = Core.ReinterpretCast(%arg0) : tensor<1x1xf16> -> tensor<1xf16>
    return %0 : tensor<1xf16>

    // CHECK: [[OUT:%.+]] = Core.ReinterpretCast([[ARG0]])
    // CHECK-SAME: -> tensor<1xf16>
    // CHECK: return [[OUT]]
}

// -----

func.func @TensorAndMemref(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    // expected-error@+1 {{Cannot change type id: 'tensor<1x1xf16>' -> 'memref<2x1xi8>'}}
    %0 = Core.ReinterpretCast(%arg0) : tensor<1x1xf16> -> memref<2x1xi8>
    %1 = Core.ReinterpretCast(%0) : memref<2x1xi8> -> tensor<1x1xf16>
    return %1 : tensor<1x1xf16>
}

// -----

func.func @AllocationSizeChange(%arg0: tensor<1x1xf16>) -> tensor<3x1xi8> {
    // expected-error@+1 {{Cannot cast to different allocation size: 'tensor<1x1xf16>' -> 'tensor<3x1xi8>'}}
    %0 = Core.ReinterpretCast(%arg0) : tensor<1x1xf16> -> tensor<3x1xi8>
    return %0 : tensor<3x1xi8>
}
