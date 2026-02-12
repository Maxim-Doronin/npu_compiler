//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-strided-dmas %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DDRType = memref<4x6xui8, @DDR>
!FlatDDRType = memref<1x24x1x1xui8, @DDR>
!FlatCMXType = memref<1x24x1x1xui8, @CMX_NN>
VPURT.SW.Runtime
    entryPoint : @VPU.SW::@runtime
    stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Multiply(memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>)
    attributes {
        VPU.kernel_code = "eltwise_mul.cpp",
        VPU.kernel_entry = "eltwise_mul",
        VPU.kernel_name = "eltwise_mul",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

net.NetworkInfo entryPoint : @LegalizeStridedDmas inputsInfo : {
    DataInfo "Parameter_1" : tensor<4x6xui8> {dynamicStrides}
    DataInfo "Parameter_2" : tensor<4x6xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<4x6xui8> {dynamicStrides}
}

// CHECK-LABEL: @LegalizeStridedDmas
func.func @LegalizeStridedDmas(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %0 = VPUIP.GenericReshape inputs(%arg0 : !DDRType) -> !FlatDDRType
    %1 = VPUIP.GenericReshape inputs(%arg1 : !DDRType) -> !FlatDDRType
    %2 = memref.alloc() : !FlatCMXType
    %3 = VPUIP.NNDMA inputs(%0 : !FlatDDRType) outputs(%2 : !FlatCMXType) -> !FlatCMXType
    %4 = memref.alloc() : !FlatCMXType
    %5 = VPUIP.NNDMA inputs(%1 : !FlatDDRType) outputs(%4 : !FlatCMXType) -> !FlatCMXType
    %6 = memref.alloc() : !FlatCMXType
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%3 as %arg3: !FlatCMXType, %5 as %arg4: !FlatCMXType) outputs(%6 as %arg5: !FlatCMXType) on tile 0 -> !FlatCMXType{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !FlatCMXType, !FlatCMXType, !FlatCMXType
    }
    %7 = VPUIP.GenericReshape inputs(%arg2 : !DDRType) -> !FlatDDRType
    %8 = VPUIP.NNDMA inputs(%results : !FlatCMXType) outputs(%7 : !FlatDDRType) -> !FlatDDRType
    return %arg2 : !DDRType

    // CHECK:   [[ALLOC_OUTPUT:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>

    // CHECK:   [[ALLOC_INPUT0:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>
    // CHECK:   [[NNDMA_0:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:  inputs(%arg0 : memref<4x6xui8, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC_INPUT0]] : memref<4x6xui8, @DDR>)

    // CHECK:   [[ALLOC_INPUT1:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>
    // CHECK:   [[NNDMA_1:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:  inputs(%arg1 : memref<4x6xui8, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC_INPUT1]] : memref<4x6xui8, @DDR>)

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = VPUIP.GenericReshape inputs([[ALLOC_OUTPUT]] : memref<4x6xui8, @DDR>) -> memref<1x24x1x1xui8, @DDR>
    // CHECK:   [[INCOMPATIBLE_OUT_DMA:%.+]] = VPUIP.NNDMA
    // CHECK:   [[CONCAT_VIEW_OUT:%.+]] = VPUIP.ConcatView inputs([[INCOMPATIBLE_OUT_DMA]] : memref<1x24x1x1xui8, @DDR>) outputs([[ALLOC_OUTPUT]]
    // CHECK:   VPUIP.NNDMA {stridedOutput}
    // CHECK-SAME:  inputs([[CONCAT_VIEW_OUT]]
    // CHECK-SAME:  outputs(%arg2 : memref<4x6xui8, @DDR>) -> memref<4x6xui8, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DDRType = memref<4x6xui8, @DDR>
!FlatDDRType = memref<1x24x1x1xui8, @DDR>
!FlatCMXType = memref<1x24x1x1xui8, @CMX_NN>
VPURT.SW.Runtime
    entryPoint : @VPU.SW::@runtime
    stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Multiply(memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>)
    attributes {
        VPU.kernel_code = "eltwise_mul.cpp",
        VPU.kernel_entry = "eltwise_mul",
        VPU.kernel_name = "eltwise_mul",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

net.NetworkInfo entryPoint : @LegalizeStridedDmasOneInput inputsInfo : {
    DataInfo "Parameter_1" : tensor<1x4x6xui8>
    DataInfo "Parameter_2" : tensor<1x4x6xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<1x4x6xui8> {dynamicStrides}
}

// CHECK-LABEL: @LegalizeStridedDmasOneInput
func.func @LegalizeStridedDmasOneInput(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %0 = VPUIP.GenericReshape inputs(%arg0 : !DDRType) -> !FlatDDRType
    %1 = VPUIP.GenericReshape inputs(%arg1 : !DDRType) -> !FlatDDRType
    %2 = memref.alloc() : !FlatCMXType
    %3 = VPUIP.NNDMA inputs(%0 : !FlatDDRType) outputs(%2 : !FlatCMXType) -> !FlatCMXType
    %4 = memref.alloc() : !FlatCMXType
    %5 = VPUIP.NNDMA inputs(%1 : !FlatDDRType) outputs(%4 : !FlatCMXType) -> !FlatCMXType
    %6 = memref.alloc() : !FlatCMXType
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%3 as %arg3: !FlatCMXType, %5 as %arg4: !FlatCMXType) outputs(%6 as %arg5: !FlatCMXType) on tile 0 -> !FlatCMXType{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !FlatCMXType, !FlatCMXType, !FlatCMXType
    }
    %7 = VPUIP.GenericReshape inputs(%arg2 : !DDRType) -> !FlatDDRType
    %8 = VPUIP.NNDMA inputs(%results : !FlatCMXType) outputs(%7 : !FlatDDRType) -> !FlatDDRType
    return %arg2 : !DDRType

    // CHECK:   [[ALLOC_OUTPUT:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>
    // CHECK:   [[ARG0_RESHAPE:%.+]] = VPUIP.GenericReshape inputs(%arg0
    // CHECK:   [[ALLOC_INPUT0:%.+]] = memref.alloc() : memref<4x6xui8, @DDR>

    // CHECK:   [[NNDMA_0:%.+]] = VPUIP.NNDMA {stridedInput}
    // CHECK-SAME:  inputs(%arg1 : memref<4x6xui8, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC_INPUT0]] : memref<4x6xui8, @DDR>)

    // CHECK-NOT: VPUIP.NNDMA {stridedInput} inputs([[ARG0_RESHAPE]]

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = VPUIP.GenericReshape inputs([[ALLOC_OUTPUT]] : memref<4x6xui8, @DDR>) -> memref<1x24x1x1xui8, @DDR>
    // CHECK:   [[INCOMPATIBLE_OUT_DMA:%.+]] = VPUIP.NNDMA
    // CHECK:   [[CONCAT_VIEW_OUT:%.+]] = VPUIP.ConcatView inputs([[INCOMPATIBLE_OUT_DMA]] : memref<1x24x1x1xui8, @DDR>) outputs([[ALLOC_OUTPUT]]
    // CHECK:   VPUIP.NNDMA {stridedOutput} inputs([[CONCAT_VIEW_OUT]] : memref<4x6xui8, @DDR>) outputs(%arg2 : memref<4x6xui8, @DDR>) -> memref<4x6xui8, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DDRType = memref<4x6xui8, @DDR>
!FlatDDRType = memref<1x24x1x1xui8, @DDR>
!FlatCMXType = memref<1x24x1x1xui8, @CMX_NN>
VPURT.SW.Runtime
    entryPoint : @VPU.SW::@runtime
    stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Multiply(memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>)
    attributes {
        VPU.kernel_code = "eltwise_mul.cpp",
        VPU.kernel_entry = "eltwise_mul",
        VPU.kernel_name = "eltwise_mul",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

net.NetworkInfo entryPoint : @NoStridedCopies inputsInfo : {
    DataInfo "Parameter_1" : tensor<1x4x6xui8>
    DataInfo "Parameter_2" : tensor<1x4x6xui8>
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<1x4x6xui8>
}

// CHECK-LABEL: @NoStridedCopies
func.func @NoStridedCopies(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRType) -> !DDRType {
    %0 = VPUIP.GenericReshape inputs(%arg0 : !DDRType) -> !FlatDDRType
    %1 = VPUIP.GenericReshape inputs(%arg1 : !DDRType) -> !FlatDDRType
    %2 = memref.alloc() : !FlatCMXType
    %3 = VPUIP.NNDMA inputs(%0 : !FlatDDRType) outputs(%2 : !FlatCMXType) -> !FlatCMXType
    %4 = memref.alloc() : !FlatCMXType
    %5 = VPUIP.NNDMA inputs(%1 : !FlatDDRType) outputs(%4 : !FlatCMXType) -> !FlatCMXType
    %6 = memref.alloc() : !FlatCMXType
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%3 as %arg3: !FlatCMXType, %5 as %arg4: !FlatCMXType) outputs(%6 as %arg5: !FlatCMXType) on tile 0 -> !FlatCMXType{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !FlatCMXType, !FlatCMXType, !FlatCMXType
    }
    %7 = VPUIP.GenericReshape inputs(%arg2 : !DDRType) -> !FlatDDRType
    %8 = VPUIP.NNDMA inputs(%results : !FlatCMXType) outputs(%7 : !FlatDDRType) -> !FlatDDRType
    return %arg2 : !DDRType

    // CHECK:       VPUIP.NNDMA
    // CHECK-NOT:       {stridedInput} inputs({{%.+}} : memref<1x4x6xui8,    @DDR>)
    // CHECK:                                 inputs({{%.+}} : memref<1x24x1x1xui8, @DDR>)

    // CHECK:       VPUIP.NNDMA
    // CHECK-NOT:       {stridedInput} inputs({{%.+}} : memref<1x4x6xui8,    @DDR>)
    // CHECK:                                 inputs({{%.+}} : memref<1x24x1x1xui8, @DDR>)

    // CHECK:       VPUIP.NNDMA
    // CHECK-NOT:       {stridedOutput} inputs({{%.*}} : memref<1x4x6xui8,    @DDR>) outputs({{%.+}} : memref<4x6xui8, @DDR>)
    // CHECK:                                  inputs({{%.*}} : memref<1x24x1x1xui8, @CMX_NN>
    // CHECK:                                  outputs({{%.+}} : memref<1x24x1x1xui8, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!DDRType = memref<1x4x64x320xf32, @DDR>
!DDRTypeOut = memref<1x4x64x320xf16, @DDR>
!CMXType = memref<1x4x64x320xf16, @CMX_NN>

VPURT.SW.Runtime
    entryPoint : @VPU.SW::@runtime
    stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Multiply(memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>)
    attributes {
        VPU.kernel_code = "eltwise_mul.cpp",
        VPU.kernel_entry = "eltwise_mul",
        VPU.kernel_name = "eltwise_mul",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

net.NetworkInfo entryPoint : @LabelStridedCopies inputsInfo : {
DataInfo "Parameter_233593" : tensor<1x4x64x320xf32> {dynamicStrides}
DataInfo "Parameter_233594" : tensor<1x4x64x320xf32> {dynamicStrides}
} outputsInfo : {
DataInfo "Add_233595" friendlyName = "Result_233596" : tensor<1x4x64x320xf32> {dynamicStrides}
}

// CHECK-LABEL: @LabelStridedCopies
func.func @LabelStridedCopies(%arg0: !DDRType, %arg1: !DDRType, %arg2: !DDRTypeOut) -> !DDRTypeOut {
    %1 = memref.alloc() : !CMXType
    %2 = VPUIP.ConvertDMA {stridedInput}  inputs(%arg0 : !DDRType) outputs(%1 : !CMXType) -> !CMXType
    %3 = memref.alloc() : !CMXType
    %4 = VPUIP.ConvertDMA {stridedInput}  inputs(%arg1 :!DDRType) outputs(%3 : !CMXType) -> !CMXType
    %5 = memref.alloc() : !CMXType
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%2 as %arg3: !CMXType, %4 as %arg4: !CMXType) outputs(%5 as %arg5: !CMXType) on tile 0 -> !CMXType{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !CMXType, !CMXType, !CMXType
    }
    %11 = VPUIP.NNDMA {stridedOutput} inputs(%results : !CMXType) outputs(%arg2 : !DDRTypeOut) -> !DDRTypeOut
    return %11 : !DDRTypeOut

    // CHECK: VPUIP.ConvertDMA {stridedInput}
    // CHECK: VPUIP.ConvertDMA {stridedInput}
    // CHECK: VPUIP.NNDMA {stridedOutput}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DDRType = memref<4x6xui8, @DDR>
!FlatDDRType = memref<1x12x1x1xui8, @DDR>
!FlatCMXType = memref<1x12x1x1xui8, @CMX_NN>

VPURT.SW.Runtime
    entryPoint : @VPU.SW::@runtime
    stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Multiply(memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>)
    attributes {
        VPU.kernel_code = "eltwise_mul.cpp",
        VPU.kernel_entry = "eltwise_mul",
        VPU.kernel_name = "eltwise_mul",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

net.NetworkInfo entryPoint : @LegalizeStridedDmasMultipleIncompatibleOutputDmas inputsInfo : {
    DataInfo "Parameter_1" : tensor<4x6xui8>
    DataInfo "Parameter_2" : tensor<4x6xui8>
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<4x6xui8> {dynamicStrides}
}

// CHECK-LABEL: @LegalizeStridedDmasMultipleIncompatibleOutputDmas
func.func @LegalizeStridedDmasMultipleIncompatibleOutputDmas(%arg0: memref<4x6xui8, @DDR>, %arg1: memref<4x6xui8, @DDR>, %arg2: memref<4x6xui8, @DDR>) -> memref<4x6xui8, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %1 = VPUIP.SubView %arg0 [2, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %2 = VPUIP.SubView %arg1 [0, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %3 = VPUIP.SubView %arg1 [2, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %4 = VPUIP.SubView %arg2 [0, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %5 = VPUIP.SubView %arg2 [2, 0] [2, 6] : memref<4x6xui8, @DDR> to memref<2x6xui8, @DDR>
    %6 = VPUIP.GenericReshape inputs(%0 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %7 = VPUIP.GenericReshape inputs(%1 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %8 = VPUIP.GenericReshape inputs(%2 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %9 = VPUIP.GenericReshape inputs(%3 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %10 = VPUIP.GenericReshape inputs(%4 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %11 = VPUIP.GenericReshape inputs(%5 : memref<2x6xui8, @DDR>) -> !FlatDDRType
    %alloc = memref.alloc() : !FlatCMXType
    %alloc_1 = memref.alloc() : !FlatCMXType
    %alloc_2 = memref.alloc() : !FlatCMXType
    %alloc_3 = memref.alloc() : !FlatCMXType
    %alloc_4 = memref.alloc() : !FlatCMXType
    %alloc_5 = memref.alloc() : !FlatCMXType
    %12 = VPUIP.NNDMA  inputs(%6 : !FlatDDRType) outputs(%alloc :!FlatCMXType) -> !FlatCMXType
    %13 = VPUIP.NNDMA  inputs(%7 : !FlatDDRType) outputs(%alloc_1 : !FlatCMXType) -> !FlatCMXType
    %14 = VPUIP.NNDMA  inputs(%8 : !FlatDDRType) outputs(%alloc_2 : !FlatCMXType) -> !FlatCMXType
    %15 = VPUIP.NNDMA  inputs(%9 : !FlatDDRType) outputs(%alloc_3 : !FlatCMXType) -> !FlatCMXType
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%12 as %arg3: !FlatCMXType, %13 as %arg4: !FlatCMXType)
                                                                                                     outputs(%alloc_4 as %arg5: !FlatCMXType) on tile 0 -> !FlatCMXType{
      VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !FlatCMXType, !FlatCMXType, !FlatCMXType
    }
    %results_10 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%14 as %arg3: !FlatCMXType, %15 as %arg4: !FlatCMXType)
                                                                                                        outputs(%alloc_5 as %arg5: !FlatCMXType) on tile 0 -> !FlatCMXType{
      VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !FlatCMXType, !FlatCMXType, !FlatCMXType
    }
    %20 = VPUIP.NNDMA  inputs(%results :!FlatCMXType) outputs(%10 : memref<1x12x1x1xui8, @DDR>) -> !FlatDDRType
    %21 = VPUIP.NNDMA  inputs(%results_10 : !FlatCMXType) outputs(%11 : memref<1x12x1x1xui8, @DDR>) -> !FlatDDRType
    return %arg2 : memref<4x6xui8, @DDR>

    // CHECK: [[OUTPUT_ALLOC:%.+]] = memref.alloc()
    // CHECK: [[CONCAT_OUTPUT:%.+]] = VPUIP.ConcatView
    // CHECK: VPUIP.NNDMA {stridedOutput} inputs([[CONCAT_OUTPUT]]
    // CHECK-SAME: outputs(%arg2
  }

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!CMXSlice = memref<2x6xui8, @CMX_NN>
!DDRSlice = memref<2x6xui8, @DDR>

VPURT.SW.Runtime
    entryPoint : @VPU.SW::@runtime
    stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Multiply(memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>, memref<*xui8, @CMX_NN>)
    attributes {
        VPU.kernel_code = "eltwise_mul.cpp",
        VPU.kernel_entry = "eltwise_mul",
        VPU.kernel_name = "eltwise_mul",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

net.NetworkInfo entryPoint : @LegalizeStridedDmasWithConcatInBetween inputsInfo : {
    DataInfo "Parameter_1" : tensor<4x6xui8> {dynamicStrides}
} outputsInfo : {
    DataInfo "Multiply_3" friendlyName = "Result_4" tensorNames = ["Multiply_Result"] : tensor<2x6xui8> {dynamicStrides}
}

// Below test case checks if a ViewOp that can be reached from function argument in 2 different ways is handled correctly
func.func @LegalizeStridedDmasWithConcatInBetween(%arg0: memref<4x6xui8, @DDR>, %arg1: !DDRSlice) -> !DDRSlice {
    %sub0 = VPUIP.SubView %arg0 [0, 0] [2, 6] : memref<4x6xui8, @DDR> to !DDRSlice
    %sub1 = VPUIP.SubView %arg0 [2, 0] [2, 6] : memref<4x6xui8, @DDR> to !DDRSlice
    %concat = VPUIP.ConcatView inputs(%sub0, %sub1 : !DDRSlice, !DDRSlice)
                               outputs(%sub0 : !DDRSlice) -> !DDRSlice
    %cmx_0 = memref.alloc() : !CMXSlice
    %cmx_1 = memref.alloc() : !CMXSlice
    %cmx_result = memref.alloc() : !CMXSlice
    %0 = VPUIP.NNDMA inputs(%concat : !DDRSlice) outputs(%cmx_0 : !CMXSlice) -> !CMXSlice
    %1 = VPUIP.NNDMA inputs(%concat : !DDRSlice) outputs(%cmx_1 : !CMXSlice) -> !CMXSlice
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
                                inputs(%cmx_0 as %arg3: !CMXSlice, %cmx_1 as %arg4: !CMXSlice)
                                outputs(%cmx_result as %arg5: !CMXSlice) on tile 0 -> !CMXSlice{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !CMXSlice, !CMXSlice, !CMXSlice
    }
    %2 = VPUIP.NNDMA inputs(%results : !CMXSlice) outputs(%arg1 :!DDRSlice) -> !DDRSlice
    return %arg1 : !DDRSlice

    // CHECK: VPUIP.NNDMA {stridedInput}  inputs(%2 : memref<2x6xui8, @DDR>) outputs(%alloc : memref<2x6xui8, @CMX_NN>) -> memref<2x6xui8, @CMX_NN>
    // CHECK: VPUIP.NNDMA {stridedInput}  inputs(%2 : memref<2x6xui8, @DDR>) outputs(%alloc_0 : memref<2x6xui8, @CMX_NN>) -> memref<2x6xui8, @CMX_NN>
    // CHECK: VPUIP.NNDMA {stridedOutput}  inputs(%results : memref<2x6xui8, @CMX_NN>) outputs(%arg1 : memref<2x6xui8, @DDR>) -> memref<2x6xui8, @DDR>
}
