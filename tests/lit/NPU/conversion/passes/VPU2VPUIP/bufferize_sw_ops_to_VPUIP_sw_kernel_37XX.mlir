//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL:  func.func @StridedSlice1Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice1Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x40x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x40x5xf16>
    return %output : tensor<3x40x40x5xf16>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [3, 40, 40, 5] [1, 1, 1, 3] : memref<3x40x40x15xf16> to memref<3x40x40x5xf16, {order = #NCHW, strides = [24000, 600, 15, 3]}>
    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x40x40x5xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<3x40x40x5xf16, {order = #NCHW, strides = [24000, 600, 15, 3]}>) outputs([[OUTPUT_BUFFER]] : memref<3x40x40x5xf16>) -> memref<3x40x40x5xf16>

    // CHECK: return [[OUTPUT]] : memref<3x40x40x5xf16>
}

// -----

// CHECK-LABEL:  func.func @StridedSlice2Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice2Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x20x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x20x5xf16>
    return %output : tensor<3x40x20x5xf16>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG]] [0, 0, 0, 0] [3, 40, 20, 5] [1, 1, 2, 3] : memref<3x40x40x15xf16> to memref<3x40x20x5xf16, {order = #NCHW, strides = [24000, 600, 30, 3]}>
    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x40x20x5xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<3x40x20x5xf16, {order = #NCHW, strides = [24000, 600, 30, 3]}>) outputs([[OUTPUT_BUFFER]] : memref<3x40x20x5xf16>) -> memref<3x40x20x5xf16>

    // CHECK: return [[OUTPUT]] : memref<3x40x20x5xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_StridedSlice(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, none, none, none, i64, i64, i64) attributes {VPU.kernel_code = "strided_slice.cpp", VPU.kernel_entry = "strided_slice", VPU.kernel_name = "strided_slice", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @StridedSlice3Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice3Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x20x20x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 2, 2, 3]} : tensor<3x40x40x15xf16> -> tensor<3x20x20x5xf16>
    return %output : tensor<3x20x20x5xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x40x40x15xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<3x40x40x15xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<3x40x40x15xf16, [@CMX_NN, 0]>) -> memref<3x40x40x15xf16, [@CMX_NN, 0]>

    // CHECK: [[STRIDESLICE_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x20x20x5xf16, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_StridedSlice inputs([[INPUT_CMX]] as {{[^:]+}}: memref<3x40x40x15xf16, [@CMX_NN, 0]>) outputs([[STRIDESLICE_BUFFER_CMX]] as {{[^:]+}}: memref<3x20x20x5xf16, [@CMX_NN, 0]>) on tile 0 -> memref<3x20x20x5xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [9223372036854775807, [0, 0, 0, 0], [3, 40, 40, 15], [1, 2, 2, 3], 1, 1, 1]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}) : memref<3x40x40x15xf16, [@CMX_NN, 0]>, memref<3x20x20x5xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x20x20x5xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<3x20x20x5xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER:%.+]] : memref<3x20x20x5xf16>) -> memref<3x20x20x5xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<3x20x20x5xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDWithOriginalShape
// CHECK-SAME:    [[INPUT_0:%.+]]: memref<1x16x180x16xf16>
// CHECK-SAME:    [[INPUT_1:%.+]]: memref<1x16x1458x2xsi32>
func.func @GatherNDWithOriginalShape(%arg0: tensor<1x16x180x16xf16>, %arg1: tensor<1x16x1458x2xsi32>) -> tensor<1x16x1458x16xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {
                batch_dims = 2 : i64, original_shape = [1, 16, 18, 10, 16]
            } : tensor<1x16x180x16xf16>, tensor<1x16x1458x2xsi32> -> tensor<1x16x1458x16xf16>

    return %0 : tensor<1x16x1458x16xf16>

    // CHECK:   [[ALLOC_DATA:%.+]] = memref.alloc() : memref<1x16x180x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_DATA:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x16x180x16xf16>) outputs([[ALLOC_DATA]] : memref<1x16x180x16xf16, [@CMX_NN, 0]>) -> memref<1x16x180x16xf16, [@CMX_NN, 0]>

    // CHECK:   [[ALLOC_INDICES:%.+]] = memref.alloc() : memref<1x16x1458x2xsi32, [@CMX_NN, 0]>
    // CHECK:   [[COPY_INDICES:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : memref<1x16x1458x2xsi32>) outputs([[ALLOC_INDICES]] : memref<1x16x1458x2xsi32, [@CMX_NN, 0]>)

    // CHECK:   [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x16x1458x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[GATHERND:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GatherND
    // CHECK-SAME:              inputs([[COPY_DATA]] as [[DATA:%.+]]: memref<1x16x180x16xf16, [@CMX_NN, 0]>, [[COPY_INDICES]] as [[INDICES:%.+]]: memref<1x16x1458x2xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:              outputs([[ALLOC_OUT]] as [[OUT:%.+]]: memref<1x16x1458x16xf16, [@CMX_NN, 0]>)
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     attrs = [2, [68719476741, 77309411338, 4294967312]]

    // Note: 68719476741: 0x_10_00000005 (16, 5); 77309411338: 0x_12_0000000a (18, 10); 4294967312: 0x_1_00000010 (1, 16)

    // CHECK:   [[ALLOC_RESULT:%.+]] = memref.alloc() : memref<1x16x1458x16xf16>
    // CHECK:   [[ALLOC_DATA:%.+]] = VPUIP.Copy inputs([[GATHERND]] : memref<1x16x1458x16xf16, [@CMX_NN, 0]>) outputs([[ALLOC_RESULT]] : memref<1x16x1458x16xf16>) -> memref<1x16x1458x16xf16>

    // CHECK: return [[ALLOC_DATA]] : memref<1x16x1458x16xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDWithoutOriginalShape
// CHECK-SAME:    [[INPUT_0:%.+]]: memref<1x16x180x16xf16>
// CHECK-SAME:    [[INPUT_1:%.+]]: memref<1x16x1458x1xsi32>
func.func @GatherNDWithoutOriginalShape(%arg0: tensor<1x16x180x16xf16>, %arg1: tensor<1x16x1458x1xsi32>) -> tensor<1x16x1458x16xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {
                batch_dims = 2 : i64
            } : tensor<1x16x180x16xf16>, tensor<1x16x1458x1xsi32> -> tensor<1x16x1458x16xf16>

    return %0 : tensor<1x16x1458x16xf16>

    // CHECK:   [[ALLOC_DATA:%.+]] = memref.alloc() : memref<1x16x180x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_DATA:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x16x180x16xf16>) outputs([[ALLOC_DATA]] : memref<1x16x180x16xf16, [@CMX_NN, 0]>) -> memref<1x16x180x16xf16, [@CMX_NN, 0]>

    // CHECK:   [[ALLOC_INDICES:%.+]] = memref.alloc() : memref<1x16x1458x1xsi32, [@CMX_NN, 0]>
    // CHECK:   [[COPY_INDICES:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : memref<1x16x1458x1xsi32>) outputs([[ALLOC_INDICES]] : memref<1x16x1458x1xsi32, [@CMX_NN, 0]>)

    // CHECK:   [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x16x1458x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[GATHERND:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GatherND
    // CHECK-SAME:              inputs([[COPY_DATA]] as [[DATA:%.+]]: memref<1x16x180x16xf16, [@CMX_NN, 0]>, [[COPY_INDICES]] as [[INDICES:%.+]]: memref<1x16x1458x1xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:              outputs([[ALLOC_OUT]] as [[OUT:%.+]]: memref<1x16x1458x16xf16, [@CMX_NN, 0]>)
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     attrs = [2, [0]]

    // CHECK:   [[ALLOC_RESULT:%.+]] = memref.alloc() : memref<1x16x1458x16xf16>
    // CHECK:   [[COPY_RESULT:%.+]] = VPUIP.Copy inputs([[GATHERND]] : memref<1x16x1458x16xf16, [@CMX_NN, 0]>) outputs([[ALLOC_RESULT]] : memref<1x16x1458x16xf16>) -> memref<1x16x1458x16xf16>

    // CHECK: return [[COPY_RESULT]] : memref<1x16x1458x16xf16>
}
