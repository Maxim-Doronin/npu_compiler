//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-memref-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

func.func private @main_func0(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>,
                              %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x90x1000x16xf16>) outputs(%arg1 : memref<1x90x1000x16xf16>)
       -> memref<1x90x1000x16xf16>
    %1 = VPUIP.Copy inputs(%arg1 : memref<1x90x1000x16xf16>) outputs(%arg2 : memref<1x90x1000x16xf16>)
       -> memref<1x90x1000x16xf16>

    return %1 : memref<1x90x1000x16xf16>
}

func.func @main(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>,
                %arg2: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> {
    %c90 = arith.constant 90 : index
    %c720 = arith.constant 720 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x720x1000x16xf16>

    scf.for %arg3 = %c0 to %c720 step %c90 {
      %subview = memref.subview %arg0[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16>
               to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16>
                 to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %0 = builtin.unrealized_conversion_cast %subview
         : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %1 = builtin.unrealized_conversion_cast %subview_0
         : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %alloc_1 = memref.alloc() : memref<1x90x1000x16xf16>
      %2 = func.call @main_func0(%0, %1, %alloc_1)
         : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
      %subview_2 = memref.subview %alloc[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
                 : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      memref.copy %2, %subview_2 : memref<1x90x1000x16xf16>
          to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
    }

    memref.copy %alloc, %arg2 : memref<1x720x1000x16xf16> to memref<1x720x1000x16xf16>
    return %arg2 : memref<1x720x1000x16xf16>
}

// CHECK: func.func private [[MAIN_FUNC0:@.+]]([[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>, [[_:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x720x1000x16xf16>, [[ARG1:%.+]]: memref<1x720x1000x16xf16>, [[ARG2:%.+]]: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> {
// CHECK:   [[C90:%.+]] = arith.constant 90 : index
// CHECK:   [[C720:%.+]] = arith.constant 720 : index
// CHECK:   [[C0:%.+]] = arith.constant 0 : index

// CHECK:   scf.for [[ARG3:%.+]] = [[C0]] to [[C720]] step [[C90]] {
// CHECK:     [[SUBVIEW0:%.+]] = memref.subview [[ARG0]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:     [[SUBVIEW1:%.+]] = memref.subview [[ARG1]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:     [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW0]]
// CHECK:     [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW1]]
// CHECK:     [[SUBVIEW2:%.+]] = memref.subview [[ARG2]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:     [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW2]]
// CHECK:     [[CALL:%.+]] = func.call [[MAIN_FUNC0]]([[CAST0]], [[CAST1]], [[CAST2]])

// -----

func.func private @main_func0(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>,
                              %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x90x1000x16xf16>) outputs(%arg1 : memref<1x90x1000x16xf16>)
       -> memref<1x90x1000x16xf16>
    %1 = VPUIP.Copy inputs(%arg1 : memref<1x90x1000x16xf16>) outputs(%arg2 : memref<1x90x1000x16xf16>)
       -> memref<1x90x1000x16xf16>

    return %1 : memref<1x90x1000x16xf16>
}

func.func @main(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>,
                %arg2: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x90x1000x16xf16>

    %call = func.call @main_func0(%arg0, %arg1, %alloc)
       : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>

    memref.copy %alloc, %arg2 : memref<1x90x1000x16xf16> to memref<1x90x1000x16xf16>
    return %arg2 : memref<1x90x1000x16xf16>
}

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x90x1000x16xf16>, [[ARG1:%.+]]: memref<1x90x1000x16xf16>, [[ARG2:%.+]]: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
// CHECK:    [[CALL:%.+]] = call @main_func0([[ARG0]], [[ARG1]], [[ARG2]])
// CHECK:    return [[ARG2]] : memref<1x90x1000x16xf16>
