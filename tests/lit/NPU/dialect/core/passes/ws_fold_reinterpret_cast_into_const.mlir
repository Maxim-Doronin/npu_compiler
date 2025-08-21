//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ws-fold-reinterpret-cast-into-const %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

{-#
  dialect_resources: {
    builtin: {
        ov_0: "0x1000000080120304"
    }
  }
#-}

// CHECK: func.func @DenseResource() -> tensor<4xi8>
func.func @DenseResource() -> tensor<4xi8> {
    %cst = const.Declare tensor<2xsi16> = dense_resource<ov_0> : tensor<2xsi16>
    // 0x8012 -> 0x80 (-128), 0x12; 0x0304 -> 0x03, 0x04
    %cast = Core.ReinterpretCast(%cst) : tensor<2xsi16> -> tensor<4xi8>
    return %cast : tensor<4xi8>

    // CHECK: [[CST:%.+]] = const.Declare tensor<4xi8> = dense<[-128, 18, 3, 4]>
    // CHECK: return [[CST]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
        ov_f32: "0x1000000043800304"
    }
  }
#-}

// CHECK: func.func @DenseResource_f32() -> tensor<12xi8>
func.func @DenseResource_f32() -> tensor<12xi8> {
    %cst = const.Declare tensor<3xf32> = dense_resource<ov_f32> : tensor<1xf32>, [#const.PadWithZero<[1], [1]>]
    // 0x43800304 -> 0x43 (67), 0x80 (-128), 0x03, 0x04
    %cast = Core.ReinterpretCast(%cst) : tensor<3xf32> -> tensor<12xi8>
    return %cast : tensor<12xi8>

    // CHECK: [[CST:%.+]] = const.Declare tensor<12xi8> = dense<[0, 0, 0, 0, 67, -128, 3, 4, 0, 0, 0, 0]>
    // CHECK: return [[CST]]
}

// -----

// CHECK: func.func private @outlined() -> tensor<3xi8>
func.func private @outlined() -> tensor<3xi8> {
    %cst = const.Declare tensor<3xsi8> = dense<[-2, 5, 3]> : tensor<3xsi8>
    %cast = Core.ReinterpretCast(%cst) : tensor<3xsi8> -> tensor<3xi8>
    return %cast : tensor<3xi8>

    // CHECK: [[CST:%.+]] = const.Declare tensor<3xi8> = dense<[-2, 5, 3]>
    // CHECK: return [[CST]]
}

// CHECK: func.func @SimpleOutlining([[IN:%.+]]: tensor<1x2x3x4xf32>)
// CHECK-SAME: -> (tensor<1x2x3x4xf32>, tensor<3xi8>, tensor<4xi8>)
func.func @SimpleOutlining(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>, tensor<3xi8>, tensor<4xi8>) {
    %out1 = func.call @outlined() : () -> tensor<3xi8>

    %cst = const.Declare tensor<4xui8> = dense<42> : tensor<2xui8>, [#const.PadWithZero<[0], [2]>]
    %out2 = Core.ReinterpretCast(%cst) : tensor<4xui8> -> tensor<4xi8>

    return %arg0, %out1, %out2 : tensor<1x2x3x4xf32>, tensor<3xi8>, tensor<4xi8>

    // CHECK: [[OUT1:%.+]] = call @outlined()
    // CHECK: [[OUT2:%.+]] = const.Declare tensor<4xi8> = dense<[42, 42, 0, 0]>
    // CHECK: return [[IN]], [[OUT1]], [[OUT2]]
}

// -----

// CHECK: func.func @Splat() -> tensor<4xi8>
func.func @Splat() -> tensor<4xi8> {
    %cst = const.Declare tensor<4xsi8> = dense<42> : tensor<4xsi8>
    %cast = Core.ReinterpretCast(%cst) : tensor<4xsi8> -> tensor<4xi8>
    return %cast : tensor<4xi8>

    // CHECK: [[CST:%.+]] = const.Declare tensor<4xi8> = dense<42>
    // CHECK: return [[CST]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
        ov_0: "0x1000000080120304"
    }
  }
#-}

func.func private @main_part1(%arg: tensor<2xsi16>) -> tensor<2xsi16> {
    return %arg : tensor<2xsi16>
}

// CHECK: func.func @MultiUserConst() -> (tensor<4xi8>, tensor<2xsi16>)
func.func @MultiUserConst() -> (tensor<4xi8>, tensor<2xsi16>) {
    %cst = const.Declare tensor<2xsi16> = dense_resource<ov_0> : tensor<2xsi16>
    // 0x8012 -> 0x80 (-128), 0x12; 0x0304 -> 0x03, 0x04
    %cast = Core.ReinterpretCast(%cst) : tensor<2xsi16> -> tensor<4xi8>
    %call = func.call @main_part1(%cst) : (tensor<2xsi16>) -> tensor<2xsi16>
    return %cast, %call : tensor<4xi8>, tensor<2xsi16>

    // CHECK: [[ORIG_CST:%.+]] = const.Declare tensor<2xsi16> = dense_resource<ov_0>
    // CHECK: [[CST:%.+]] = const.Declare tensor<4xi8> = dense<[-128, 18, 3, 4]>
    // CHECK: [[CALL:%.+]] = call @main_part1([[ORIG_CST]])
    // CHECK: return [[CST]], [[CALL]]
}
