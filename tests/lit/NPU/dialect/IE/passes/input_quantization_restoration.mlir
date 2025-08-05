//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --input-quantization-restoration %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// CHECK-LABEL: @InputQuantizationRestoration4D
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @InputQuantizationRestoration4D(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x1600x2560xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    return %1 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.997686803> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[FAKEQUANTIZE]] : tensor<1x4x1600x2560xf32>
}

// -----

// CHECK-LABEL: @InputQuantizationRestoration3D
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x300x560xui8>
func.func @InputQuantizationRestoration3D(%arg0: tensor<1x300x560xui8>) -> tensor<1x300x560xf32> {
    %scale = const.Declare tensor<1x1x1xf32> = dense<-1.0> : tensor<1x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x300x560xui8> -> tensor<1x300x560xf32>
    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x300x560xf32>, tensor<1x1x1xf32> -> tensor<1x300x560xf32>
    return %1 : tensor<1x300x560xf32>

    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-2.550000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x300x560xui8> -> tensor<1x300x560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x300x560xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x300x560xf32>
    // CHECK: return [[FAKEQUANTIZE]] : tensor<1x300x560xf32>
}

// -----

// CHECK-LABEL: @ConvertAsSecondMultiplyInput
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @ConvertAsSecondMultiplyInput(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x1600x2560xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Multiply(%scale, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x4x1600x2560xf32> -> tensor<1x4x1600x2560xf32>
    return %1 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.997686803> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[FAKEQUANTIZE]] : tensor<1x4x1600x2560xf32>
}

// -----

// CHECK-LABEL: @NoInputQuantizationRestorationI8
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xi8>
func.func @NoInputQuantizationRestorationI8(%arg0: tensor<1x4x1600x2560xi8>) -> tensor<1x4x1600x2560xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xi8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    return %1 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xi8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x4x1600x2560xf32>
}

// -----

// CHECK-LABEL: @NoInputQuantizationRestorationDiffChannel
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @NoInputQuantizationRestorationDiffChannel(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x1600x2560xf32> {
    %cst = const.Declare tensor<1x4x1x1xf32> = dense<70.5> : tensor<1x4x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x1600x2560xf32>
    return %1 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x4x1x1xf32> = dense<7.050000e+01> : tensor<1x4x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x4x1600x2560xf32>
}
