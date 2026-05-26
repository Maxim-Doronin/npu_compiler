//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --replace-allocs-with-single-alloc-and-views --canonicalize --cse --verify-diagnostics %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

func.func @StaticShapeAllocs(%arg0: memref<1x1024xf16>, %arg1: memref<1x1024xf16>) -> memref<1x1024xf16> {
    %alloc = memref.alloc() : memref<1x1024xf16>
    memref.copy %arg0, %alloc : memref<1x1024xf16> to memref<1x1024xf16>

    %alloc1 = memref.alloc() : memref<1x1024xf16>
    memref.copy %alloc, %alloc1 : memref<1x1024xf16> to memref<1x1024xf16>

    memref.copy %alloc1, %arg1 : memref<1x1024xf16> to memref<1x1024xf16>

    return %arg1 : memref<1x1024xf16>
}

// CHECK: func.func @StaticShapeAllocs([[ARG0:%.+]]: memref<1x1024xf16>, [[ARG1:%.+]]: memref<1x1024xf16>) -> memref<1x1024xf16> {
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[C2048:%.+]] = arith.constant 2048 : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc() {alignment = 64 : i64} : memref<4096xi8>
// CHECK:   [[VIEW:%.+]] = memref.view [[ALLOC]][[[C0]]][] : memref<4096xi8> to memref<1x1024xf16>
// CHECK:   [[VIEW_0:%.+]] = memref.view [[ALLOC]][[[C2048]]][] : memref<4096xi8> to memref<1x1024xf16>
// CHECK:   memref.copy [[ARG0]], [[VIEW]] : memref<1x1024xf16> to memref<1x1024xf16>
// CHECK:   memref.copy [[VIEW]], [[VIEW_0]] : memref<1x1024xf16> to memref<1x1024xf16>
// CHECK:   memref.copy [[VIEW_0]], [[ARG1]] : memref<1x1024xf16> to memref<1x1024xf16>

// CHECK:   return [[ARG1]] : memref<1x1024xf16>

// -----

func.func @DynamicShapeNoAlloc(%arg0: memref<1x?x?xf16>, %arg1: memref<1x?x?xf16>) -> memref<1x?x?xf16> {
    memref.copy %arg0, %arg1 : memref<1x?x?xf16> to memref<1x?x?xf16>

    return %arg1 : memref<1x?x?xf16>
    // CHECK-NOT: memref.alloc
}

// -----

func.func @DynamicShapeSingleAlloc(%arg0: memref<1x?x?xf16>, %arg1: memref<1x?x?xf16>) -> memref<1x?x?xf16> {
    %c1 = arith.constant 1 : index
    %dim1 = memref.dim %arg0, %c1 : memref<1x?x?xf16>

    %c2 = arith.constant 2 : index
    %dim2 = memref.dim %arg0, %c2 : memref<1x?x?xf16>

    %alloc = memref.alloc(%dim1, %dim2) : memref<1x?x?xf16>
    memref.copy %arg0, %alloc : memref<1x?x?xf16> to memref<1x?x?xf16>

    memref.copy %alloc, %arg1 : memref<1x?x?xf16> to memref<1x?x?xf16>

    return %arg1 : memref<1x?x?xf16>
}

// CHECK: func.func @DynamicShapeSingleAlloc([[ARG0:%.+]]: memref<1x?x?xf16>, [[ARG1:%.+]]: memref<1x?x?xf16>) -> memref<1x?x?xf16> {
// CHECK-DAG:   [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:   [[C2:%.+]] = arith.constant 2 : index

// CHECK:   [[DIM1:%.+]] = memref.dim [[ARG0]], [[C1]] : memref<1x?x?xf16>
// CHECK:   [[DIM2:%.+]] = memref.dim [[ARG0]], [[C2]] : memref<1x?x?xf16>
// CHECK:   [[MULTIPLY1:%.+]] = arith.muli [[DIM1]], [[DIM2]] : index
// CHECK:   [[MULTIPLY2:%.+]] = arith.muli [[MULTIPLY1]], [[C2]] : index

// CHECK:   [[ALLOC:%.+]] = memref.alloc([[MULTIPLY2]]) {alignment = 64 : i64} : memref<?xi8>
// CHECK:   [[VIEW:%.+]] = memref.view [[ALLOC]][[[C0]]][[[DIM1]], [[DIM2]]] : memref<?xi8> to memref<1x?x?xf16>

// CHECK:   memref.copy [[ARG0]], [[VIEW]] : memref<1x?x?xf16> to memref<1x?x?xf16>
// CHECK:   memref.copy [[VIEW]], [[ARG1]] : memref<1x?x?xf16> to memref<1x?x?xf16>

// CHECK:   return [[ARG1]] : memref<1x?x?xf16>

// -----

func.func @AllocWithSubviews(%arg0: memref<1x16x720x?xf16>, %arg1: memref<1x16x720x40xf16>, %arg2: index) -> memref<1x16x720x40xf16> {
    %c3 = arith.constant 3 : index

    %dim = memref.dim %arg0, %c3 : memref<1x16x720x?xf16>
    %alloc = memref.alloc(%dim) : memref<1x16x720x?xf16>

    %subview = memref.subview %arg0[0, 0, 0, %arg2] [1, 16, 720, 40] [1, 1, 1, 1]
             : memref<1x16x720x?xf16> to memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[0, 0, 0, %arg2] [1, 16, 720, 40] [1, 1, 1, 1]
               : memref<1x16x720x?xf16> to memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>>
    memref.copy %subview, %subview_2 : memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>>
    memref.copy %subview_2, %arg1 : memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x720x40xf16>

    return %arg1 : memref<1x16x720x40xf16>
}

// CHECK: func.func @AllocWithSubviews([[ARG0:%.+]]: memref<1x16x720x?xf16>, [[ARG1:%.+]]: memref<1x16x720x40xf16>, [[ARG2:%.+]]: index) -> memref<1x16x720x40xf16> {
// CHECK:   [[C23040:%.+]] = arith.constant 23040 : index
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[C3:%.+]] = arith.constant 3 : index

// CHECK:   [[DIM:%.+]] = memref.dim [[ARG0]], [[C3]] : memref<1x16x720x?xf16>
// CHECK:   [[MULTIPLY:%.+]] = arith.muli [[DIM]], [[C23040]] : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc([[MULTIPLY]]) {alignment = 64 : i64} : memref<?xi8>
// CHECK:   [[VIEW:%.+]] = memref.view [[ALLOC]][[[C0]]][[[DIM]]] : memref<?xi8> to memref<1x16x720x?xf16>
// CHECK:   [[SUBVIEW:%.+]] = memref.subview [[ARG0]][0, 0, 0, [[ARG2]]] [1, 16, 720, 40] [1, 1, 1, 1]
// CHECK:   [[SUBVIEW_0:%.+]] = memref.subview [[VIEW]][0, 0, 0, [[ARG2]]] [1, 16, 720, 40] [1, 1, 1, 1]
// CHECK:   memref.copy [[SUBVIEW]], [[SUBVIEW_0]]
// CHECK:   memref.copy [[SUBVIEW_0]], [[ARG1]]
// CHECK:   return [[ARG1]]

// -----

// expected-error@+1 {{ReplaceAllocsWithSingleAllocAndViewsPass cannot be applied to functions with dealloc operations}}
func.func @AllocDeallocInvalid() {
    %alloc = memref.alloc() : memref<1x16x720x40xf16>
    memref.dealloc %alloc : memref<1x16x720x40xf16>
    return
}


// -----

func.func @NestedAlloc(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x4x4xf16>, %flag: i1) -> memref<1x16x4x4xf16> {

    scf.if %flag {
        %alloc0 = memref.alloc() : memref<1x16x4x4xf16>
        memref.copy %arg0, %alloc0 : memref<1x16x4x4xf16> to memref<1x16x4x4xf16>
        memref.copy %alloc0, %arg1 : memref<1x16x4x4xf16> to memref<1x16x4x4xf16>
    } else {
        %alloc2 = memref.alloc() : memref<1x16x4x4xf16>
        memref.copy %arg0, %alloc2 : memref<1x16x4x4xf16> to memref<1x16x4x4xf16>
        memref.copy %alloc2, %arg1 : memref<1x16x4x4xf16> to memref<1x16x4x4xf16>
    }

    return %arg1 : memref<1x16x4x4xf16>
}

// CHECK: func.func @NestedAlloc([[ARG0:%.+]]: memref<1x16x4x4xf16>, [[ARG1:%.+]]: memref<1x16x4x4xf16>, [[ARG2:%.+]]: i1) -> memref<1x16x4x4xf16> {
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[C512:%.+]] = arith.constant 512 : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc() {alignment = 64 : i64} : memref<1024xi8>
// CHECK:   [[VIEW_0:%.+]] = memref.view [[ALLOC]][[[C0]]][] : memref<1024xi8> to memref<1x16x4x4xf16>
// CHECK:   [[VIEW_1:%.+]] = memref.view [[ALLOC]][[[C512]]][] : memref<1024xi8> to memref<1x16x4x4xf16>
// CHECK:   scf.if [[ARG2]] {
// CHECK:     memref.copy [[ARG0]], [[VIEW_0]] : memref<1x16x4x4xf16> to memref<1x16x4x4xf16>
// CHECK:     memref.copy [[VIEW_0]], [[ARG1]] : memref<1x16x4x4xf16> to memref<1x16x4x4xf16>
// CHECK:   } else {
// CHECK:     memref.copy [[ARG0]], [[VIEW_1]] : memref<1x16x4x4xf16> to memref<1x16x4x4xf16>
// CHECK:     memref.copy [[VIEW_1]], [[ARG1]] : memref<1x16x4x4xf16> to memref<1x16x4x4xf16>
// CHECK:   }
// CHECK:   return [[ARG1]]

// -----

func.func @NestedDynAlloc(%arg0: memref<1x16x?x4xf16>, %arg1: memref<1x16x?x4xf16>, %flag: i1) -> memref<1x16x?x4xf16> {
    %c2 = arith.constant 2 : index loc(unknown)
    %arg_c0 = memref.dim %arg0, %c2 : memref<1x16x?x4xf16>
    scf.if %flag {
        %alloc0 = memref.alloc(%arg_c0) : memref<1x16x?x4xf16>
        memref.copy %arg0, %alloc0 : memref<1x16x?x4xf16> to memref<1x16x?x4xf16>
        memref.copy %alloc0, %arg1 : memref<1x16x?x4xf16> to memref<1x16x?x4xf16>
    } else {
        %alloc2 = memref.alloc(%arg_c0) : memref<1x16x?x4xf16>
        memref.copy %arg0, %alloc2 : memref<1x16x?x4xf16> to memref<1x16x?x4xf16>
        memref.copy %alloc2, %arg1 : memref<1x16x?x4xf16> to memref<1x16x?x4xf16>
    }

    return %arg1 : memref<1x16x?x4xf16>
}

// CHECK: func.func @NestedDynAlloc([[ARG0:%.+]]: memref<1x16x?x4xf16>, [[ARG1:%.+]]: memref<1x16x?x4xf16>, [[ARG2:%.+]]: i1) -> memref<1x16x?x4xf16> {
// CHECK:   [[C128:%.+]]  = arith.constant 128 : index
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[DYN_DIM_IDX:%.+]] = arith.constant 2 : index
// CHECK:   [[DYN_DIM:%.+]] = memref.dim [[ARG0]], [[DYN_DIM_IDX]] : memref<1x16x?x4xf16>
// CHECK:   [[ALLOC_OFFSET:%.+]] = arith.muli [[DYN_DIM]], [[C128]] : index
// CHECK:   [[ALLOC_SIZE:%.+]] = arith.addi [[ALLOC_OFFSET]], [[ALLOC_OFFSET]] : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc([[ALLOC_SIZE]]) {alignment = 64 : i64} : memref<?xi8>
// CHECK:   [[VIEW_0:%.+]] = memref.view [[ALLOC]][[[C0]]][[[DYN_DIM]]] : memref<?xi8> to memref<1x16x?x4xf16>
// CHECK:   [[VIEW_1:%.+]] = memref.view [[ALLOC]][[[ALLOC_OFFSET]]][[[DYN_DIM]]]  : memref<?xi8> to memref<1x16x?x4xf16>
// CHECK:   scf.if [[ARG2]] {
// CHECK:     memref.copy [[ARG0]], [[VIEW_0]] : memref<1x16x?x4xf16> to memref<1x16x?x4xf16>
// CHECK:     memref.copy [[VIEW_0]], [[ARG1]] : memref<1x16x?x4xf16> to memref<1x16x?x4xf16>
// CHECK:   } else {
// CHECK:     memref.copy [[ARG0]], [[VIEW_1]] : memref<1x16x?x4xf16> to memref<1x16x?x4xf16>
// CHECK:     memref.copy [[VIEW_1]], [[ARG1]] : memref<1x16x?x4xf16> to memref<1x16x?x4xf16>
// CHECK:   }
// CHECK:   return [[ARG1]]

// -----

func.func @main_batching1(%i: memref<1x3x?x?xf32>, %j: memref<1x3x?x?xf32>) -> memref<1x3x?x?xf32> {
    return %i: memref<1x3x?x?xf32>
}

func.func @scf_alloc_for_batch(%input: memref<?x3x?x?xf32>, %main: memref<?x3x?x?xf32>) -> memref<?x3x?x?xf32> attributes {config.pureHostCompileFunc} {
    %c2 = arith.constant 2 : index loc(unknown)
    %c3 = arith.constant 3 : index loc(unknown)
    %c1 = arith.constant 1 : index loc(unknown)
    %c0 = arith.constant 0 : index loc(unknown)
    %main_c0 = memref.dim %input, %c0 : memref<?x3x?x?xf32>
    %input_c2 = memref.dim %input, %c2 : memref<?x3x?x?xf32>
    %input_c21 = arith.addi %input_c2, %c1 : index
    %input_c21_div = arith.divsi %input_c21, %c2 : index
    %input_c21_div_mul = arith.muli %input_c21_div, %c2 : index
    %input_c3 = memref.dim %input, %c3 : memref<?x3x?x?xf32>
    %input_c31 = arith.addi %input_c3, %c1 : index
    %input_c31_div = arith.divsi %input_c31, %c2 : index
    %input_c31_div_mul = arith.muli %input_c31_div, %c2 : index
    scf.for %iter = %c0 to %main_c0 step %c1 {
        %main_7 = memref.subview %input[%iter, 0, 0, 0] [1, 3, %input_c2, %input_c3] [1, 1, 1, 1] : memref<?x3x?x?xf32> to memref<1x3x?x?xf32, strided<[?, ?, ?, 1], offset: ?>>
        %main_8 = memref.alloc(%input_c2, %input_c3) {alignment = 64 : i64} : memref<1x3x?x?xf32>
        memref.copy %main_7, %main_8 : memref<1x3x?x?xf32, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x?x?xf32>
        %main_9 = memref.subview %main[%iter, 0, 0, 0] [1, 3, %input_c21_div_mul, %input_c31_div_mul] [1, 1, 1, 1] : memref<?x3x?x?xf32> to memref<1x3x?x?xf32, strided<[?, ?, ?, 1], offset: ?>>
        %main_10 = memref.cast %main_9 : memref<1x3x?x?xf32, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x?x?xf32>
        %main_11 = func.call @main_batching1(%main_8, %main_10) : (memref<1x3x?x?xf32>, memref<1x3x?x?xf32>) -> memref<1x3x?x?xf32>
    }
    return %main : memref<?x3x?x?xf32>
}


// CHECK: func.func @scf_alloc_for_batch([[INPUT:%.+]]: memref<?x3x?x?xf32>, [[OUTPUT:%.+]]: memref<?x3x?x?xf32>) -> memref<?x3x?x?xf32>
// CHECK:   [[C_4:%.+]] = arith.constant 4 : index
// CHECK:   [[C_2:%.+]] = arith.constant 2 : index
// CHECK:   [[C_3:%.+]] = arith.constant 3 : index
// CHECK:   [[C_1:%.+]] = arith.constant 1 : index
// CHECK:   [[C_0:%.+]] = arith.constant 0 : index
// CHECK:   [[END:%.+]] = memref.dim [[INPUT]], [[C_0]] : memref<?x3x?x?xf32>
// CHECK:   [[DIM_2:%.+]] = memref.dim [[INPUT]], [[C_2]] : memref<?x3x?x?xf32>
// CHECK:   [[DIM_3:%.+]] = memref.dim [[INPUT]], [[C_3]] : memref<?x3x?x?xf32>
// CHECK:   [[DIM_OFFSET_2:%.+]] = arith.muli [[DIM_2]], [[C_3]] : index
// CHECK:   [[DIM_OFFSET_23:%.+]] = arith.muli [[DIM_OFFSET_2]], [[DIM_3]] : index
// CHECK:   [[DIM_OFFSET_234:%.+]]  = arith.muli [[DIM_OFFSET_23]], [[C_4]] : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc([[DIM_OFFSET_234]]) {alignment = 64 : i64} : memref<?xi8>
// CHECK:   [[VIEW:%.+]] = memref.view [[ALLOC]][[[C_0]]][[[DIM_2]], [[DIM_3]]] : memref<?xi8> to memref<1x3x?x?xf32>
// CHECK:   scf.for [[ITER:%.+]] = [[C_0]] to [[END]] step [[C_1]] {
// CHECK:     [[SUBVIEW:%.+]] = memref.subview [[INPUT]][[[ITER]], 0, 0, 0] [1, 3, [[DIM_2]], [[DIM_3]]] [1, 1, 1, 1] : memref<?x3x?x?xf32> to memref<1x3x?x?xf32, strided<[?, ?, ?, 1], offset: ?>>
// CHECK:     memref.copy [[SUBVIEW]], [[VIEW]] : memref<1x3x?x?xf32, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x?x?xf32>
// CHECK:  return [[OUTPUT]] : memref<?x3x?x?xf32>
