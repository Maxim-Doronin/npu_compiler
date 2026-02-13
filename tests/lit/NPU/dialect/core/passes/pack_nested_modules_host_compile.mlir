//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=HostCompile" --pack-nested-modules %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @DynamicStridesCopyInputOutput {
// CHECK-COUNT-1:  net.NetworkInfo entryPoint : @main inputsInfo : {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "in_0" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "out_0" : tensor<1x3x60x60xf16>
  }

// CHECK-LABEL:  module @Module0 attributes {{.+}} {
// CHECK:  net.NetworkInfo entryPoint : @main_part1 inputsInfo : {
// CHECK:    DataInfo "in_0" : tensor<1x3x60x60xf16> {dynamicStrides}
// CHECK:    DataInfo "in_1" : tensor<1x3x60x60xf16> {dynamicStrides}
// CHECK:  } outputsInfo : {
// CHECK:    DataInfo "out_0" : tensor<1x3x60x60xf16> {dynamicStrides}


  func.func private @main_part1(%arg0: memref<1x3x60x60xf16> {func.dynamicStrides = true}, %arg1: memref<1x3x60x60xf16> {func.dynamicStrides = true}) -> (memref<1x3x60x60xf16> {func.dynamicStrides = true}){
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %0 : memref<1x3x60x60xf16>
  }

  func.func @main(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %alloc = memref.alloc() : memref<1x3x60x60xf16>
    %0 = func.call @main_part1(%arg0, %alloc) : (memref<1x3x60x60xf16>, memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %arg1 : memref<1x3x60x60xf16>
  }
}
