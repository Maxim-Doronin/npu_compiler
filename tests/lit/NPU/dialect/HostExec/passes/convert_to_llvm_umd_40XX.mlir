//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --mlir-elide-elementsattrs-if-larger 8 --convert-to-llvm-umd-calls %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @OneInputOneOutput
module @OneInputOneOutput {

// CHECK-DAG: llvm.func @[[npu_level_zero_alloc:.+]](i64, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @[[npu_level_zero_append_memory_copy:.+]](!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
// CHECK:  func.func @main([[arg0:%.+]]: memref<1x3x60x60xf16>, [[arg1:%.+]]: memref<1x3x60x60xf16>

  func.func @main(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %alloc = memref.alloc() : memref<1x3x60x60xf16>
    memref.copy %alloc, %arg1 : memref<1x3x60x60xf16> to memref<1x3x60x60xf16>
    return %arg1 : memref<1x3x60x60xf16>

// CHECK:    [[op0:%.+]] = builtin.unrealized_conversion_cast [[arg1]] : memref<1x3x60x60xf16> to !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:    [[op1:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:    [[op2:%.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-DAG:    [[op3:%.+]] = llvm.mlir.constant(60 : index) : i64
// CHECK-DAG:    [[op4:%.+]] = llvm.mlir.constant(60 : index) : i64
// CHECK-DAG:    [[op5:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:    [[op6:%.+]] = llvm.mlir.constant(3600 : index) : i64
// CHECK-DAG:    [[op7:%.+]] = llvm.mlir.constant(10800 : index) : i64
// CHECK-DAG:    [[op8:%.+]] = llvm.mlir.constant(10800 : index) : i64
// CHECK-DAG:    [[op9:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:    [[op10:%.+]] = llvm.getelementptr [[op9]][[[op8]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-NEXT:    [[op11:%.+]] = llvm.ptrtoint [[op10]] : !llvm.ptr to i64
// CHECK-NEXT:    [[op12:%.+]] = llvm.call @[[npu_level_zero_alloc]]([[op11]], %arg2) : (i64, !llvm.ptr) -> !llvm.ptr
// CHECK:    [[op13:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op14:%.+]] = llvm.insertvalue [[op12]], [[op13]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op15:%.+]] = llvm.insertvalue [[op12]], [[op14]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:    [[op16:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    [[op17:%.+]] = llvm.insertvalue [[op16]], [[op15]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op18:%.+]] = llvm.insertvalue [[op1]], [[op17]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op19:%.+]] = llvm.insertvalue [[op2]], [[op18]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op20:%.+]] = llvm.insertvalue [[op3]], [[op19]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op21:%.+]] = llvm.insertvalue [[op4]], [[op20]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op22:%.+]] = llvm.insertvalue [[op7]], [[op21]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op23:%.+]] = llvm.insertvalue [[op6]], [[op22]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op24:%.+]] = llvm.insertvalue [[op4]], [[op23]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NEXT:    [[op25:%.+]] = llvm.insertvalue [[op5]], [[op24]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:    [[op26:%.+]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-DAG:    [[op27:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:    [[op28:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    [[op29:%.+]] = llvm.alloca [[op28]] x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store [[op25]], [[op29]] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
// CHECK:    [[op30:%.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK-NEXT:    [[op31:%.+]] = llvm.insertvalue [[op27]], [[op30]][0] : !llvm.struct<(i64, ptr)>
// CHECK-NEXT:    [[op32:%.+]] = llvm.insertvalue [[op29]], [[op31]][1] : !llvm.struct<(i64, ptr)>
// CHECK-DAG:    [[op33:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:    [[op34:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    [[op35:%.+]] = llvm.alloca [[op34]] x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store [[op0]], [[op35]] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
// CHECK:    [[op36:%.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK-NEXT:    [[op37:%.+]] = llvm.insertvalue [[op33]], [[op36]][0] : !llvm.struct<(i64, ptr)>
// CHECK-NEXT:    [[op38:%.+]] = llvm.insertvalue [[op35]], [[op37]][1] : !llvm.struct<(i64, ptr)>
// CHECK:    [[op39:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    [[op40:%.+]] = llvm.alloca [[op39]] x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
// CHECK-DAG:    llvm.store [[op32]], [[op40]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
// CHECK-DAG:    [[op41:%.+]] = llvm.alloca [[op39]] x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store [[op38]], [[op41]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
// CHECK:    [[op42:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    [[op43:%.+]] = llvm.getelementptr [[op42]][1] : (!llvm.ptr) -> !llvm.ptr, f16
// CHECK-NEXT:    [[op44:%.+]] = llvm.ptrtoint [[op43]] : !llvm.ptr to i64
// CHECK-NEXT:    llvm.call @npu_level_zero_append_memory_copy([[op40]], [[op41]], [[op44]], %arg5) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.intr.stackrestore [[op26]] : !llvm.ptr
 }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @StaticEltwiseNHWC attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x720x1000xf16>
        DataInfo "input2" : tensor<1x16x720x1000xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x16x720x1000xf16>
    }

  module @Module0 {
    func.func private @main_func0(%arg0: memref<1x16x90x1000xf16>, %arg1: memref<1x16x90x1000xf16>, %arg2: memref<1x16x90x1000xf16>) -> memref<1x16x90x1000xf16> {
      %7 = VPUIP.Copy inputs(%arg0 : memref<1x16x90x1000xf16>) outputs(%arg2 : memref<1x16x90x1000xf16>) -> memref<1x16x90x1000xf16>
      return %7 : memref<1x16x90x1000xf16>
    }
  }
  func.func @main(%arg0: memref<1x16x720x1000xf16>, %arg1: memref<1x16x720x1000xf16>, %arg2: memref<1x16x720x1000xf16>) -> memref<1x16x720x1000xf16> {
    %c90 = llvm.mlir.constant(90 : index) : i64
    %c720 = llvm.mlir.constant(720 : index) : i64
    %c0 = llvm.mlir.constant(0 : index) : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16x720x1000xf16>
    memref.copy %arg1, %alloc : memref<1x16x720x1000xf16> to memref<1x16x720x1000xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x16x720x1000xf16>
    memref.copy %arg0, %alloc_0 : memref<1x16x720x1000xf16> to memref<1x16x720x1000xf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x16x720x1000xf16>
    %0 = llvm.icmp "slt" %c720, %c0 : i64
    %i1 = llvm.sdiv %c720, %c90 : i64
    %1 = builtin.unrealized_conversion_cast %i1 : i64 to index
    %c0i = builtin.unrealized_conversion_cast %c0 : i64 to index
    %c720i = builtin.unrealized_conversion_cast %c720 : i64 to index
    %c90i = builtin.unrealized_conversion_cast %c90 : i64 to index
    %2 = async.create_group %1 : !async.group
    scf.for %count = %c0i to %c720i step %c90i {
      %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<1x16x90x1000xf16>
      %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<1x16x90x1000xf16>
      %alloc_3 = memref.alloc() : memref<1x16x90x1000xf16>
      %token, %bodyResults = async.execute -> !async.value<memref<1x16x90x1000xf16>> {
        %8 = Core.NestedCall @Module0::@main_func0(%alloc1, %alloc2, %alloc_3) : (memref<1x16x90x1000xf16>, memref<1x16x90x1000xf16>, memref<1x16x90x1000xf16>) -> memref<1x16x90x1000xf16>
        async.yield %8 : memref<1x16x90x1000xf16>
      }
      %6 = async.add_to_group %token, %2 : !async.token
      %7 = async.await %bodyResults : !async.value<memref<1x16x90x1000xf16>>
    }
    async.await_all %2
    %3 = VPUIP.Copy inputs(%alloc_1 : memref<1x16x720x1000xf16>) outputs(%arg2 : memref<1x16x720x1000xf16>) -> memref<1x16x720x1000xf16>
    return %3 : memref<1x16x720x1000xf16>
  }
  //CHECK-NOT: async.create_group
  //CHECK-NOT: async.add_to_group
  //CHECK-NOT: async.await_all
  //CHECK-NOT: async.await
  //CHECK: llvm.call @npu_level_zero_reset_commandlist
  //CHECK: llvm.call @npu_level_zero_submit_commandlist
}
