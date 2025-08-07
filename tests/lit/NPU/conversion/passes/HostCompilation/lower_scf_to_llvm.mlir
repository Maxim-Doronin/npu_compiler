//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --hostexec-to-llvm %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module {
  // @main_part1_kernel is filled with random values as this LIT test is not supposed to use this for inference
  llvm.mlir.global internal constant @main_part1_kernel("\7FELF\02\01\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00") {addr_space = 0 : i32}
  func.func @main(%arg0: memref<1x3x?x60xf32>, %arg1: memref<1x3x?x60xf32>, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(128520 : i64) : i64
    %1 = llvm.mlir.constant(3 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(60 : index) : i64
    %5 = builtin.unrealized_conversion_cast %arg0 : memref<1x3x?x60xf32> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %6 = builtin.unrealized_conversion_cast %arg1 : memref<1x3x?x60xf32> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %7 = builtin.unrealized_conversion_cast %4 : i64 to index
    %8 = builtin.unrealized_conversion_cast %3 : i64 to index
    %9 = llvm.extractvalue %5[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %11 = llvm.mul %9, %4  : i64
    %12 = llvm.mul %11, %1  : i64
    %13 = llvm.mul %12, %2  : i64
    %14 = llvm.mlir.zero : !llvm.ptr
    %15 = llvm.getelementptr %14[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.call @npu_level_zero_alloc(%16, %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
    %18 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %17, %19[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %3, %20[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %2, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %1, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.insertvalue %9, %23[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.insertvalue %4, %24[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.insertvalue %12, %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.insertvalue %11, %26[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %4, %27[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.insertvalue %2, %28[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = builtin.unrealized_conversion_cast %29 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x3x?x60xf16>
    %31 = scf.for %arg11 = %8 to %10 step %7 iter_args(%arg12 = %30) -> (memref<1x3x?x60xf16>) {
      %68 = builtin.unrealized_conversion_cast %arg12 : memref<1x3x?x60xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %subview = memref.subview %arg0[0, 0, %arg11, 0] [1, 3, 60, 60] [1, 1, 1, 1] : memref<1x3x?x60xf32> to memref<1x3x60x60xf32, strided<[?, ?, 60, 1], offset: ?>>
      %69 = builtin.unrealized_conversion_cast %subview : memref<1x3x60x60xf32, strided<[?, ?, 60, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %70 = llvm.mlir.zero : !llvm.ptr
      %71 = llvm.getelementptr %70[10800] : (!llvm.ptr) -> !llvm.ptr, f32
      %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
      %73 = llvm.call @npu_level_zero_alloc(%72, %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
      %74 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %75 = llvm.extractvalue %69[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %76 = llvm.mul %75, %2  : i64
      %77 = llvm.extractvalue %69[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %78 = llvm.mul %76, %77  : i64
      %79 = llvm.extractvalue %69[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %80 = llvm.mul %78, %79  : i64
      %81 = llvm.extractvalue %69[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %82 = llvm.mul %80, %81  : i64
      %83 = llvm.mlir.zero : !llvm.ptr
      %84 = llvm.getelementptr %83[1] : (!llvm.ptr) -> !llvm.ptr, f32
      %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
      %86 = llvm.mul %82, %85  : i64
      llvm.call @npu_level_zero_append_memory_copy(%74, %73, %86, %arg5) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      %87 = llvm.mlir.zero : !llvm.ptr
      %88 = llvm.getelementptr %87[10800] : (!llvm.ptr) -> !llvm.ptr, f16
      %89 = llvm.ptrtoint %88 : !llvm.ptr to i64
      %90 = llvm.call @npu_level_zero_alloc(%89, %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
      %91 = llvm.mlir.addressof @main_part1_kernel : !llvm.ptr
      %92 = llvm.getelementptr %91[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128520 x i8>
      llvm.call @npu_level_zero_launch(%73, %90, %92, %0, %arg2, %arg3, %arg4, %arg5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
      %93 = llvm.extractvalue %68[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %94 = llvm.mul %93, %4  : i64
      %95 = llvm.mul %94, %1  : i64
      %96 = llvm.mul %95, %2  : i64
      %97 = llvm.mlir.zero : !llvm.ptr
      %98 = llvm.getelementptr %97[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      %99 = llvm.ptrtoint %98 : !llvm.ptr to i64
      %100 = llvm.call @npu_level_zero_alloc(%99, %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
      %101 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %102 = llvm.insertvalue %100, %101[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %103 = llvm.insertvalue %100, %102[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %104 = llvm.insertvalue %3, %103[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %105 = llvm.insertvalue %2, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %106 = llvm.insertvalue %1, %105[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %107 = llvm.insertvalue %93, %106[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %108 = llvm.insertvalue %4, %107[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %109 = llvm.insertvalue %95, %108[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %110 = llvm.insertvalue %94, %109[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %111 = llvm.insertvalue %4, %110[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %112 = llvm.insertvalue %2, %111[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %113 = builtin.unrealized_conversion_cast %112 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x3x?x60xf16>
      %114 = llvm.extractvalue %68[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %115 = llvm.extractvalue %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %116 = llvm.mul %115, %2  : i64
      %117 = llvm.extractvalue %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %118 = llvm.mul %116, %117  : i64
      %119 = llvm.extractvalue %68[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %120 = llvm.mul %118, %119  : i64
      %121 = llvm.extractvalue %68[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %122 = llvm.mul %120, %121  : i64
      %123 = llvm.mlir.zero : !llvm.ptr
      %124 = llvm.getelementptr %123[1] : (!llvm.ptr) -> !llvm.ptr, f16
      %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
      %126 = llvm.mul %122, %125  : i64
      llvm.call @npu_level_zero_append_memory_copy(%114, %100, %126, %arg5) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      %subview_0 = memref.subview %113[0, 0, %arg11, 0] [1, 3, 60, 60] [1, 1, 1, 1] : memref<1x3x?x60xf16> to memref<1x3x60x60xf16, strided<[?, ?, 60, 1], offset: ?>>
      %127 = builtin.unrealized_conversion_cast %subview_0 : memref<1x3x60x60xf16, strided<[?, ?, 60, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %128 = llvm.extractvalue %127[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %129 = llvm.mul %2, %2  : i64
      %130 = llvm.mul %129, %1  : i64
      %131 = llvm.mul %130, %4  : i64
      %132 = llvm.mul %131, %4  : i64
      %133 = llvm.mlir.zero : !llvm.ptr
      %134 = llvm.getelementptr %133[1] : (!llvm.ptr) -> !llvm.ptr, f16
      %135 = llvm.ptrtoint %134 : !llvm.ptr to i64
      %136 = llvm.mul %132, %135  : i64
      llvm.call @npu_level_zero_append_memory_copy(%90, %128, %136, %arg5) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      scf.yield %113 : memref<1x3x?x60xf16>
    }
    llvm.call @npu_level_zero_append_barrier(%arg5) : (!llvm.ptr) -> ()
    %32 = llvm.mul %9, %4  : i64
    %33 = llvm.mul %32, %1  : i64
    %34 = llvm.mul %33, %2  : i64
    %35 = llvm.mlir.zero : !llvm.ptr
    %36 = llvm.getelementptr %35[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.call @npu_level_zero_alloc(%37, %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
    %39 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %41 = llvm.insertvalue %38, %40[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.insertvalue %3, %41[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.insertvalue %2, %42[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.insertvalue %1, %43[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.insertvalue %9, %44[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.insertvalue %4, %45[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.insertvalue %33, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.insertvalue %32, %47[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.insertvalue %4, %48[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.insertvalue %2, %49[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %51 = builtin.unrealized_conversion_cast %50 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x3x?x60xf32>
    %52 = scf.for %arg11 = %8 to %10 step %7 iter_args(%arg12 = %51) -> (memref<1x3x?x60xf32>) {
      %68 = builtin.unrealized_conversion_cast %arg12 : memref<1x3x?x60xf32> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %subview = memref.subview %31[0, 0, %arg11, 0] [1, 3, 60, 60] [1, 1, 1, 1] : memref<1x3x?x60xf16> to memref<1x3x60x60xf16, strided<[?, ?, 60, 1], offset: ?>>
      %69 = builtin.unrealized_conversion_cast %subview : memref<1x3x60x60xf16, strided<[?, ?, 60, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %70 = llvm.mlir.zero : !llvm.ptr
      %71 = llvm.getelementptr %70[10800] : (!llvm.ptr) -> !llvm.ptr, f16
      %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
      %73 = llvm.call @npu_level_zero_alloc(%72, %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
      %74 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %75 = llvm.extractvalue %69[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %76 = llvm.mul %75, %2  : i64
      %77 = llvm.extractvalue %69[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %78 = llvm.mul %76, %77  : i64
      %79 = llvm.extractvalue %69[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %80 = llvm.mul %78, %79  : i64
      %81 = llvm.extractvalue %69[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %82 = llvm.mul %80, %81  : i64
      %83 = llvm.mlir.zero : !llvm.ptr
      %84 = llvm.getelementptr %83[1] : (!llvm.ptr) -> !llvm.ptr, f16
      %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
      %86 = llvm.mul %82, %85  : i64
      llvm.call @npu_level_zero_append_memory_copy(%74, %73, %86, %arg5) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      %87 = llvm.mlir.zero : !llvm.ptr
      %88 = llvm.getelementptr %87[10800] : (!llvm.ptr) -> !llvm.ptr, f32
      %89 = llvm.ptrtoint %88 : !llvm.ptr to i64
      %90 = llvm.call @npu_level_zero_alloc(%89, %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
      %91 = llvm.mlir.addressof @main_part1_kernel : !llvm.ptr
      %92 = llvm.getelementptr %91[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128520 x i8>
      llvm.call @npu_level_zero_launch(%73, %90, %92, %0, %arg2, %arg3, %arg4, %arg5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
      %93 = llvm.extractvalue %68[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %94 = llvm.mul %93, %4  : i64
      %95 = llvm.mul %94, %1  : i64
      %96 = llvm.mul %95, %2  : i64
      %97 = llvm.mlir.zero : !llvm.ptr
      %98 = llvm.getelementptr %97[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %99 = llvm.ptrtoint %98 : !llvm.ptr to i64
      %100 = llvm.call @npu_level_zero_alloc(%99, %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
      %101 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %102 = llvm.insertvalue %100, %101[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %103 = llvm.insertvalue %100, %102[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %104 = llvm.insertvalue %3, %103[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %105 = llvm.insertvalue %2, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %106 = llvm.insertvalue %1, %105[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %107 = llvm.insertvalue %93, %106[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %108 = llvm.insertvalue %4, %107[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %109 = llvm.insertvalue %95, %108[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %110 = llvm.insertvalue %94, %109[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %111 = llvm.insertvalue %4, %110[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %112 = llvm.insertvalue %2, %111[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %113 = builtin.unrealized_conversion_cast %112 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x3x?x60xf32>
      %114 = llvm.extractvalue %68[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %115 = llvm.extractvalue %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %116 = llvm.mul %115, %2  : i64
      %117 = llvm.extractvalue %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %118 = llvm.mul %116, %117  : i64
      %119 = llvm.extractvalue %68[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %120 = llvm.mul %118, %119  : i64
      %121 = llvm.extractvalue %68[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %122 = llvm.mul %120, %121  : i64
      %123 = llvm.mlir.zero : !llvm.ptr
      %124 = llvm.getelementptr %123[1] : (!llvm.ptr) -> !llvm.ptr, f32
      %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
      %126 = llvm.mul %122, %125  : i64
      llvm.call @npu_level_zero_append_memory_copy(%114, %100, %126, %arg5) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      %subview_0 = memref.subview %113[0, 0, %arg11, 0] [1, 3, 60, 60] [1, 1, 1, 1] : memref<1x3x?x60xf32> to memref<1x3x60x60xf32, strided<[?, ?, 60, 1], offset: ?>>
      %127 = builtin.unrealized_conversion_cast %subview_0 : memref<1x3x60x60xf32, strided<[?, ?, 60, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
      %128 = llvm.extractvalue %127[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
      %129 = llvm.mul %2, %2  : i64
      %130 = llvm.mul %129, %1  : i64
      %131 = llvm.mul %130, %4  : i64
      %132 = llvm.mul %131, %4  : i64
      %133 = llvm.mlir.zero : !llvm.ptr
      %134 = llvm.getelementptr %133[1] : (!llvm.ptr) -> !llvm.ptr, f32
      %135 = llvm.ptrtoint %134 : !llvm.ptr to i64
      %136 = llvm.mul %132, %135  : i64
      llvm.call @npu_level_zero_append_memory_copy(%90, %128, %136, %arg5) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      scf.yield %113 : memref<1x3x?x60xf32>
    }
    %53 = builtin.unrealized_conversion_cast %52 : memref<1x3x?x60xf32> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.call @npu_level_zero_append_barrier(%arg5) : (!llvm.ptr) -> ()
    %54 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.extractvalue %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.mul %56, %2  : i64
    %58 = llvm.extractvalue %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.mul %57, %58  : i64
    %60 = llvm.extractvalue %53[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.mul %59, %60  : i64
    %62 = llvm.extractvalue %53[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %63 = llvm.mul %61, %62  : i64
    %64 = llvm.mlir.zero : !llvm.ptr
    %65 = llvm.getelementptr %64[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %66 = llvm.ptrtoint %65 : !llvm.ptr to i64
    %67 = llvm.mul %63, %66  : i64
    llvm.call @npu_level_zero_append_memory_copy(%54, %55, %67, %arg5) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    return
  }
  llvm.func @npu_level_zero_alloc(i64, !llvm.ptr) -> !llvm.ptr
  llvm.func @npu_level_zero_append_memory_copy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
  llvm.func @npu_level_zero_launch(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @npu_level_zero_append_barrier(!llvm.ptr)
}

// CHECK-NOT: scf.for
// CHECK: llvm.br
