//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --pass-pipeline="builtin.module(builtin.module(set-memory-space{memory-space=DDR set-memory-space-for-function-boundaries=false}))" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

//CHECK-LABEL: @NestedFunction
module @NestedFunction {
  //CHECK-LABEL: @Module0
  module @Module0  {
    //CHECK-LABEL: func.func @main_func0
    //CHECK-SAME: ([[SUBMAIN_ARG0:%.+]]: memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16> {
    func.func @main_func0(%arg0: memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16> {
      %alloc = memref.alloc() : memref<1x89x1000x16xf16>
      //CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x89x1000x16xf16, @DDR>
      %alloc2 = memref.alloc() : memref<1x89x1000x16xf16>
      //CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x89x1000x16xf16, @DDR>
      %0 = VPUIP.Copy inputs(%alloc2 : memref<1x89x1000x16xf16>) outputs(%alloc : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      //CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ALLOC1]] : memref<1x89x1000x16xf16, @DDR>) outputs([[ALLOC0]] : memref<1x89x1000x16xf16, @DDR>) -> memref<1x89x1000x16xf16, @DDR>
      %1 = VPUIP.Copy inputs(%0 : memref<1x89x1000x16xf16>) outputs(%arg0 : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      //CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[COPY0]] : memref<1x89x1000x16xf16, @DDR>) outputs([[SUBMAIN_ARG0]] : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      return %1 : memref<1x89x1000x16xf16>
      //CHECK: return [[COPY1]] : memref<1x89x1000x16xf16>
    }
  }

  //CHECK-LABEL: func.func @main
  //CHECK-SAME: ([[MAIN_ARG0:%.+]]: memref<1x89x1000x16xf16>) {
  func.func @main(%arg0 : memref<1x89x1000x16xf16>) {
    %0 = Core.NestedCall @Module0::@main_func0(%arg0) : (memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
    //CHECK: Core.NestedCall @Module0::@main_func0([[MAIN_ARG0]]) : (memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
    return
  }
}

// -----

// CHECK-LABEL: @TopModule
module @TopModule {
    // CHECK-LABEL: @NestedModule
    module @NestedModule {
        // CHECK-LABEL: func.func @StridedLayoutPreservedOnMemSpaceAssignment
        // CHECK-SAME: ([[INPUT:%.+]]: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>,
        // CHECK-SAME: [[OUTPUT:%.+]]: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
        func.func @StridedLayoutPreservedOnMemSpaceAssignment(
                %input: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>,
                %output: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>> {
            %buf = memref.alloc() : memref<1x16x48x640xf16>
            // CHECK: [[BUF:%.+]] = memref.alloc() : memref<1x16x48x640xf16, @DDR>
            %copy_to_buf = VPUIP.Copy inputs(%input: memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                                      outputs(%buf : memref<1x16x48x640xf16>) -> memref<1x16x48x640xf16>
            // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
            // CHECK-SAME: outputs([[BUF]] : memref<1x16x48x640xf16, @DDR>) -> memref<1x16x48x640xf16, @DDR>

            %reinterp = Core.ReinterpretCast(%copy_to_buf) : memref<1x16x48x640xf16>
                      -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[REINTERP:%.+]] = Core.ReinterpretCast([[COPY0]]) : memref<1x16x48x640xf16, @DDR>
            // CHECK-SAME: -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>
            %last_copy = VPUIP.Copy inputs(%reinterp : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                                    outputs(%output : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
                       -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[REINTERP]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>, @DDR>)
            // CHECK-SAME: outputs([[OUTPUT]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>)
            // CHECK-SAME: -> memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>

            return %last_copy : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: return [[COPY1]] : memref<1x16x48x640xf16, strided<[?, ?, ?, ?], offset: ?>>
        }
    }
}
