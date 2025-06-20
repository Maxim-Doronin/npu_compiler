
//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-reflect-pad-to-cmx %s | FileCheck %s
// REQUIRES: arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @moveReflectPadToCmxW
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x32x32x3x!qElemType>)
func.func @moveReflectPadToCmxW(%arg0: memref<1x32x32x3x!qElemType>) -> memref<1x32x32x5x!qElemType, #NHWC, @DDR> {
    %alloc_cmx = memref.alloc() : memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>
    %copy_0 = VPUIP.Copy inputs (%arg0: memref<1x32x32x3x!qElemType>)
                        outputs (%alloc_cmx: memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>

    %alloc_ddr = memref.alloc() : memref<1x32x32x3x!qElemType, #NHWC, @DDR>
    %input_copy_to_ddr = VPUIP.Copy inputs (%copy_0: memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>)
                        outputs (%alloc_ddr: memref<1x32x32x3x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x3x!qElemType, #NHWC, @DDR>
    
    %input_pad_0 = VPUIP.SubView %input_copy_to_ddr [0, 0, 0, 1] [1, 32, 32, 1] : memref<1x32x32x3x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @DDR>
    %output_pad_alloc_0 = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_0 = VPUIP.Copy inputs(%input_pad_0 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @DDR>)
                            outputs(%output_pad_alloc_0 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x1x!qElemType, #NHWC, @DDR>
                            
    %input_pad_1 = VPUIP.SubView %input_copy_to_ddr [0, 0, 0, 2] [1, 32, 32, 1] : memref<1x32x32x3x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @DDR>
    %output_pad_alloc_1 = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_1 = VPUIP.Copy inputs(%input_pad_1 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @DDR>)
                            outputs(%output_pad_alloc_1 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    
    %concat_view_output = memref.alloc() : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
    %out_subview_0 = VPUIP.SubView %concat_view_output [0, 0, 0, 0] [1, 32, 32, 1] : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %out_copy_0 = VPUIP.Copy inputs(%input_pad_copy_0 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_0 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>)
                            -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %out_subview_1 = VPUIP.SubView %concat_view_output [0, 0, 0, 1] [1, 32, 32, 3] : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %input_copy = VPUIP.Copy inputs(%input_copy_to_ddr : memref<1x32x32x3x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_1 : memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>)
                            -> memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %out_subview_2 = VPUIP.SubView %concat_view_output [0, 0, 0, 4] [1, 32, 32, 1] : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %out_copy_1 = VPUIP.Copy inputs(%input_pad_copy_1: memref<1x32x32x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_2 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>)
                            -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    
    %concat_view = VPUIP.ConcatView inputs(%out_copy_0, %input_copy, %out_copy_1 : 
                                    memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>, 
                                    memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>, 
                                    memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>) 
                                outputs(%concat_view_output : memref<1x32x32x5x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x5x!qElemType, #NHWC, @DDR>
   
    return %concat_view: memref<1x32x32x5x!qElemType, #NHWC, @DDR>

    // CHECK:   [[ALLOC_CMX_0:%.+]] = memref.alloc() : memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x32x32x3x!qElemType>) outputs([[ALLOC_CMX_0]] : memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[ALLOC_DDR_0:%.+]] = memref.alloc() : memref<1x32x32x3x!qElemType, #NHWC, @DDR>
    // CHECK:   [[COPY_1:%.+]] = VPUIP.Copy inputs([[COPY_0]] : memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>) outputs([[ALLOC_DDR_0]] : memref<1x32x32x3x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x3x!qElemType, #NHWC, @DDR>
    
    // CHECK:   [[ALLOC_CMX_1:%.+]] = memref.alloc() : memref<1x32x32x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PAD_INPUT:%.+]] = VPUIP.Copy inputs([[COPY_1]] : memref<1x32x32x3x!qElemType, #NHWC, @DDR>) outputs([[ALLOC_CMX_1]] : memref<1x32x32x3x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x32x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    
    // CHECK:   [[IN_PAD_0:%.+]] = VPUIP.SubView [[PAD_INPUT]] [0, 0, 0, 1] [1, 32, 32, 1] : memref<1x32x32x3x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_ALLOC_0:%.+]] = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_COPY_0:%.+]] = VPUIP.Copy inputs([[IN_PAD_0]] : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, [@CMX_NN, 0]>) outputs([[IN_PAD_ALLOC_0]] : memref<1x32x32x1x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x32x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_1:%.+]] = VPUIP.SubView [[PAD_INPUT]] [0, 0, 0, 2] [1, 32, 32, 1] : memref<1x32x32x3x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_ALLOC_1:%.+]] = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_COPY_1:%.+]] = VPUIP.Copy inputs([[IN_PAD_1]] : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, [@CMX_NN, 0]>) outputs([[IN_PAD_ALLOC_1]] : memref<1x32x32x1x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x32x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    
    // CHECK:   [[CONCAT_VIEW_OUTPUT:%.+]] = memref.alloc() : memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[CONCAT_VIEW_OUTPUT_SV_0:%.+]] = VPUIP.SubView %alloc_4 [0, 0, 0, 0] [1, 32, 32, 1] : memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_COPY_0:%.+]] = VPUIP.Copy inputs([[IN_PAD_COPY_0]] : memref<1x32x32x1x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT_SV_0]] : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>) -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[CONCAT_VIEW_OUTPUT_SV_1:%.+]] = VPUIP.SubView %alloc_4 [0, 0, 0, 1] [1, 32, 32, 3] : memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[PAD_INPUT]] : memref<1x32x32x3x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT_SV_1]] : memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>) -> memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[CONCAT_VIEW_OUTPUT_SV_2:%.+]] = VPUIP.SubView %alloc_4 [0, 0, 0, 4] [1, 32, 32, 1] : memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_COPY_1:%.+]] = VPUIP.Copy inputs([[IN_PAD_COPY_1]] : memref<1x32x32x1x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT_SV_2]] : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>) -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>
    
    // CHECK:   [[CONCAT_VIEW:%.+]] = VPUIP.ConcatView inputs([[OUT_COPY_0]], [[INPUT_COPY]], [[OUT_COPY_1]] : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>, memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>, memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT]] : memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]>
    
    // CHECK:   [[CV_OUTPUT_ALLOC_DDR:%.+]] = memref.alloc() : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
    // CHECK:   [[CV_OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW]] : memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[CV_OUTPUT_ALLOC_DDR]] : memref<1x32x32x5x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x5x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[CV_OUTPUT_COPY]] : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @moveReflectPadToCmxH
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x32x3x5x!qElemType>)
func.func @moveReflectPadToCmxH(%arg0: memref<1x32x3x5x!qElemType>) -> memref<1x32x5x5x!qElemType, #NHWC, @DDR> {
    %alloc_cmx = memref.alloc() : memref<1x32x3x5x!qElemType, #NHWC, @CMX_NN>
    %copy_0 = VPUIP.Copy inputs (%arg0: memref<1x32x3x5x!qElemType>)
                        outputs (%alloc_cmx: memref<1x32x3x5x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x3x5x!qElemType, #NHWC, @CMX_NN>

    %alloc_ddr = memref.alloc() : memref<1x32x3x5x!qElemType, #NHWC, @DDR>
    %input_copy_to_ddr = VPUIP.Copy inputs (%copy_0: memref<1x32x3x5x!qElemType, #NHWC, @CMX_NN>)
                                outputs (%alloc_ddr: memref<1x32x3x5x!qElemType, #NHWC, @DDR>) -> memref<1x32x3x5x!qElemType, #NHWC, @DDR>

    %input_pad_0 = VPUIP.SubView %input_copy_to_ddr [0, 0, 1, 0] [1, 32, 1, 5] : memref<1x32x3x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [480, 1, 160, 32]}, @DDR>
    %output_pad_alloc_0 = memref.alloc() : memref<1x32x1x5x!qElemType, #NHWC, @DDR>
    %input_pad_copy_0 = VPUIP.Copy inputs(%input_pad_0 : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [480, 1, 160, 32]}, @DDR>)
                                outputs(%output_pad_alloc_0 : memref<1x32x1x5x!qElemType, #NHWC, @DDR>) -> memref<1x32x1x5x!qElemType, #NHWC, @DDR>

    %input_pad_1 = VPUIP.SubView %input_copy_to_ddr [0, 0, 2, 0] [1, 32, 1, 5] : memref<1x32x3x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [480, 1, 160, 32]}, @DDR>
    %output_pad_alloc_1 = memref.alloc() : memref<1x32x1x5x!qElemType, #NHWC, @DDR>
    %input_pad_copy_1 = VPUIP.Copy inputs(%input_pad_1 : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [480, 1, 160, 32]}, @DDR>)
                                outputs(%output_pad_alloc_1 : memref<1x32x1x5x!qElemType, #NHWC, @DDR>) -> memref<1x32x1x5x!qElemType, #NHWC, @DDR>
    
    %concat_view_output = memref.alloc() : memref<1x32x5x5x!qElemType, #NHWC, @DDR>
    %out_subview_0 = VPUIP.SubView %concat_view_output [0, 0, 0, 0] [1, 32, 1, 5] : memref<1x32x5x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>
    %out_copy_0 = VPUIP.Copy inputs(%input_pad_copy_0 : memref<1x32x1x5x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_0 : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>) -> memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>
    %out_subview_1 = VPUIP.SubView %concat_view_output [0, 0, 1, 0] [1, 32, 3, 5] : memref<1x32x5x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x3x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>
    %input_copy = VPUIP.Copy inputs(%input_copy_to_ddr : memref<1x32x3x5x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_1 : memref<1x32x3x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>) -> memref<1x32x3x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>
    %out_subview_2 = VPUIP.SubView %concat_view_output [0, 0, 4, 0] [1, 32, 1, 5] : memref<1x32x5x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>
    %out_copy_1 = VPUIP.Copy inputs(%input_pad_copy_1 : memref<1x32x1x5x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_2 : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>) -> memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>
    
    %concat_view = VPUIP.ConcatView inputs(%out_copy_0, %input_copy, %out_copy_1 :
                                    memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>,
                                    memref<1x32x3x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>,
                                    memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, @DDR>)
                                outputs(%concat_view_output : memref<1x32x5x5x!qElemType, #NHWC, @DDR>) ->  memref<1x32x5x5x!qElemType, #NHWC, @DDR>

    return %concat_view: memref<1x32x5x5x!qElemType, #NHWC, @DDR>

    // CHECK:   [[ALLOC_CMX_0:%.+]] = memref.alloc() : memref<1x32x3x5x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x32x3x5x!qElemType>) outputs([[ALLOC_CMX_0]] : memref<1x32x3x5x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x3x5x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[ALLOC_DDR_0:%.+]] = memref.alloc() : memref<1x32x3x5x!qElemType, #NHWC, @DDR>
    // CHECK:   [[COPY_1:%.+]] = VPUIP.Copy inputs([[COPY_0]] : memref<1x32x3x5x!qElemType, #NHWC, @CMX_NN>) outputs([[ALLOC_DDR_0]] : memref<1x32x3x5x!qElemType, #NHWC, @DDR>) -> memref<1x32x3x5x!qElemType, #NHWC, @DDR>

    // CHECK:   [[ALLOC_CMX_1:%.+]] = memref.alloc() : memref<1x32x3x5x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PAD_INPUT:%.+]] = VPUIP.Copy inputs([[COPY_1]] : memref<1x32x3x5x!qElemType, #NHWC, @DDR>) outputs([[ALLOC_CMX_1]] : memref<1x32x3x5x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x3x5x!qElemType, #NHWC, [@CMX_NN, 0]>

    // CHECK:   [[IN_PAD_0:%.+]] = VPUIP.SubView [[PAD_INPUT]] [0, 0, 1, 0] [1, 32, 1, 5] : memref<1x32x3x5x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [480, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_ALLOC_0:%.+]] = memref.alloc() : memref<1x32x1x5x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_COPY_0:%.+]] = VPUIP.Copy inputs([[IN_PAD_0]] : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [480, 1, 160, 32]}, [@CMX_NN, 0]>) outputs([[IN_PAD_ALLOC_0]] : memref<1x32x1x5x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x1x5x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_1:%.+]] = VPUIP.SubView [[PAD_INPUT]] [0, 0, 2, 0] [1, 32, 1, 5] : memref<1x32x3x5x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [480, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_ALLOC_1:%.+]] = memref.alloc() : memref<1x32x1x5x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_PAD_COPY_1:%.+]] = VPUIP.Copy inputs([[IN_PAD_1]] : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [480, 1, 160, 32]}, [@CMX_NN, 0]>) outputs([[IN_PAD_ALLOC_1]] : memref<1x32x1x5x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x1x5x!qElemType, #NHWC, [@CMX_NN, 0]>

    // CHECK:   [[CONCAT_VIEW_OUTPUT:%.+]] = memref.alloc() : memref<1x32x5x5x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[CONCAT_VIEW_OUTPUT_SV_0:%.+]] = VPUIP.SubView [[CONCAT_VIEW_OUTPUT]] [0, 0, 0, 0] [1, 32, 1, 5] : memref<1x32x5x5x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_COPY_0:%.+]] = VPUIP.Copy inputs([[IN_PAD_COPY_0]] : memref<1x32x1x5x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT_SV_0]] : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>) -> memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[CONCAT_VIEW_OUTPUT_SV_1:%.+]] = VPUIP.SubView [[CONCAT_VIEW_OUTPUT]] [0, 0, 1, 0] [1, 32, 3, 5] : memref<1x32x5x5x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x3x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[PAD_INPUT]] : memref<1x32x3x5x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT_SV_1]] : memref<1x32x3x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>) -> memref<1x32x3x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[CONCAT_VIEW_OUTPUT_SV_2:%.+]] = VPUIP.SubView [[CONCAT_VIEW_OUTPUT]] [0, 0, 4, 0] [1, 32, 1, 5] : memref<1x32x5x5x!qElemType, #NHWC, [@CMX_NN, 0]> to memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_COPY_1:%.+]] = VPUIP.Copy inputs([[IN_PAD_COPY_1]] : memref<1x32x1x5x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT_SV_2]] : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>) -> memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>

    // CHECK:   [[CONCAT_VIEW:%.+]] = VPUIP.ConcatView inputs([[OUT_COPY_0]], [[INPUT_COPY]], [[OUT_COPY_1]] : memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>, memref<1x32x3x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>, memref<1x32x1x5x!qElemType, {order = #NHWC, strides = [800, 1, 160, 32]}, [@CMX_NN, 0]>) outputs(%alloc_4 : memref<1x32x5x5x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x5x5x!qElemType, #NHWC, [@CMX_NN, 0]>

    // CHECK:   [[CV_OUTPUT_ALLOC_DDR:%.+]] = memref.alloc() : memref<1x32x5x5x!qElemType, #NHWC, @DDR>
    // CHECK:   [[CV_OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW]] : memref<1x32x5x5x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[CV_OUTPUT_ALLOC_DDR]] : memref<1x32x5x5x!qElemType, #NHWC, @DDR>) -> memref<1x32x5x5x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[CV_OUTPUT_COPY]] : memref<1x32x5x5x!qElemType, #NHWC, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @dontMoveReflectPadToCmxNoCmxToDdrCopy
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x32x32x3x!qElemType>)
func.func @dontMoveReflectPadToCmxNoCmxToDdrCopy(%arg0: memref<1x32x32x3x!qElemType>) -> memref<1x32x32x5x!qElemType, #NHWC, @DDR> {
    %alloc_cmx = memref.alloc() : memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>
    %copy_0 = VPUIP.Copy inputs (%arg0: memref<1x32x32x3x!qElemType>)
                        outputs (%alloc_cmx: memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>
  
    %input_pad_0 = VPUIP.SubView %copy_0 [0, 0, 0, 1] [1, 32, 32, 1] : memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @CMX_NN>
    %output_pad_alloc_0 = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_0 = VPUIP.Copy inputs(%input_pad_0 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @CMX_NN>)
                            outputs(%output_pad_alloc_0 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x1x!qElemType, #NHWC, @DDR>
                            
    %input_pad_1 = VPUIP.SubView %copy_0 [0, 0, 0, 2] [1, 32, 32, 1] : memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @CMX_NN>
    %output_pad_alloc_1 = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_1 = VPUIP.Copy inputs(%input_pad_1 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @CMX_NN>)
                            outputs(%output_pad_alloc_1 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    
    %concat_view_output = memref.alloc() : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
    %out_subview_0 = VPUIP.SubView %concat_view_output [0, 0, 0, 0] [1, 32, 32, 1] : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %out_copy_0 = VPUIP.Copy inputs(%input_pad_copy_0 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_0 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>)
                            -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %out_subview_1 = VPUIP.SubView %concat_view_output [0, 0, 0, 1] [1, 32, 32, 3] : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %input_copy = VPUIP.Copy inputs(%copy_0 : memref<1x32x32x3x!qElemType, #NHWC,@CMX_NN>)
                            outputs(%out_subview_1 : memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>)
                            -> memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %out_subview_2 = VPUIP.SubView %concat_view_output [0, 0, 0, 4] [1, 32, 32, 1] : memref<1x32x32x5x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    %out_copy_1 = VPUIP.Copy inputs(%input_pad_copy_1: memref<1x32x32x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_2 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>)
                            -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>
    
    %concat_view = VPUIP.ConcatView inputs(%out_copy_0, %input_copy, %out_copy_1 : 
                                    memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>, 
                                    memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>, 
                                    memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, @DDR>) 
                                outputs(%concat_view_output : memref<1x32x32x5x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x5x!qElemType, #NHWC, @DDR>
   
    return %concat_view: memref<1x32x32x5x!qElemType, #NHWC, @DDR>

    // CHECK:    VPUIP.ConcatView
    // CHECK-NOT:   memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>, memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>, memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [5120, 1, 160, 32]}, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT:%.+]] : memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @dontMoveReflectPadToCmxPaddingNotEnoughMemCopied
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x32x32x32x!qElemType>)
func.func @dontMoveReflectPadToCmxPaddingNotEnoughMemCopied(%arg0: memref<1x32x32x32x!qElemType>) -> memref<1x32x32x34x!qElemType, #NHWC, @DDR> {
    %alloc_cmx = memref.alloc() : memref<1x32x32x32x!qElemType, #NHWC, @CMX_NN>
    %copy_0 = VPUIP.Copy inputs (%arg0: memref<1x32x32x32x!qElemType>)
                        outputs (%alloc_cmx: memref<1x32x32x32x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x32x32x!qElemType, #NHWC, @CMX_NN>
  
    %alloc_ddr = memref.alloc() : memref<1x32x32x32x!qElemType, #NHWC, @DDR>
    %input_copy_to_ddr = VPUIP.Copy inputs (%copy_0: memref<1x32x32x32x!qElemType, #NHWC, @CMX_NN>)
                        outputs (%alloc_ddr: memref<1x32x32x32x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x32x!qElemType, #NHWC, @DDR>
   
    %input_pad_0 = VPUIP.SubView %input_copy_to_ddr [0, 0, 0, 1] [1, 32, 32, 1] : memref<1x32x32x32x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @DDR>
    %output_pad_alloc_0 = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_0 = VPUIP.Copy inputs(%input_pad_0 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @DDR>)
                            outputs(%output_pad_alloc_0 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x1x!qElemType, #NHWC, @DDR>
                            
    %input_pad_1 = VPUIP.SubView %input_copy_to_ddr [0, 0, 0, 31] [1, 32, 32, 1] : memref<1x32x32x32x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @DDR>
    %output_pad_alloc_1 = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_1 = VPUIP.Copy inputs(%input_pad_1 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @DDR>)
                            outputs(%output_pad_alloc_1 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    
    %concat_view_output = memref.alloc() : memref<1x32x32x34x!qElemType, #NHWC, @DDR>
    %out_subview_0 = VPUIP.SubView %concat_view_output [0, 0, 0, 0] [1, 32, 32, 1] : memref<1x32x32x34x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>
    %out_copy_0 = VPUIP.Copy inputs(%input_pad_copy_0 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_0 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>)
                            -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>
    %out_subview_1 = VPUIP.SubView %concat_view_output [0, 0, 0, 1] [1, 32, 32, 32] : memref<1x32x32x34x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x32x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>
    %input_copy = VPUIP.Copy inputs(%input_copy_to_ddr : memref<1x32x32x32x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_1 : memref<1x32x32x32x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>)
                            -> memref<1x32x32x32x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>
    %out_subview_2 = VPUIP.SubView %concat_view_output [0, 0, 0, 33] [1, 32, 32, 1] : memref<1x32x32x34x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>
    %out_copy_1 = VPUIP.Copy inputs(%input_pad_copy_1: memref<1x32x32x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_2 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>)
                            -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>
    
    %concat_view = VPUIP.ConcatView inputs(%out_copy_0, %input_copy, %out_copy_1 : 
                                    memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>, 
                                    memref<1x32x32x32x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>, 
                                    memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, @DDR>) 
                                outputs(%concat_view_output : memref<1x32x32x34x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x34x!qElemType, #NHWC, @DDR>
   
    return %concat_view: memref<1x32x32x34x!qElemType, #NHWC, @DDR>

    // CHECK:    VPUIP.ConcatView
    // CHECK-NOT:   memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, [@CMX_NN, 0]>, memref<1x32x32x32x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, [@CMX_NN, 0]>, memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT:%.+]] : memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x32x5x!qElemType, #NHWC, [@CMX_NN, 0]>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @dontMoveReflectPadToCmxPaddingWithMoreThan1
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x32x32x3x!qElemType>)
func.func @dontMoveReflectPadToCmxPaddingWithMoreThan1(%arg0: memref<1x32x32x3x!qElemType>) -> memref<1x32x32x6x!qElemType, #NHWC, @DDR> {
    %alloc_cmx = memref.alloc() : memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>
    %copy_0 = VPUIP.Copy inputs (%arg0: memref<1x32x32x3x!qElemType>)
                        outputs (%alloc_cmx: memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>

    %alloc_ddr = memref.alloc() : memref<1x32x32x3x!qElemType, #NHWC, @DDR>
    %input_copy_to_ddr = VPUIP.Copy inputs (%copy_0: memref<1x32x32x3x!qElemType, #NHWC, @CMX_NN>)
                        outputs (%alloc_ddr: memref<1x32x32x3x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x3x!qElemType, #NHWC, @DDR>
    
    %input_pad_0 = VPUIP.SubView %input_copy_to_ddr [0, 0, 0, 1] [1, 32, 32, 1] : memref<1x32x32x3x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @DDR>
    %output_pad_alloc_0 = memref.alloc() : memref<1x32x32x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_0 = VPUIP.Copy inputs(%input_pad_0 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @DDR>)
                            outputs(%output_pad_alloc_0 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x1x!qElemType, #NHWC, @DDR>
                            
    %input_pad_1 = VPUIP.SubView %input_copy_to_ddr [0, 0, 0, 1] [1, 32, 32, 2] : memref<1x32x32x3x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x2x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @DDR>
    %output_pad_alloc_1 = memref.alloc() : memref<1x32x32x2x!qElemType, #NHWC, @DDR>
    %input_pad_copy_1 = VPUIP.Copy inputs(%input_pad_1 : memref<1x32x32x2x!qElemType, {order = #NHWC, strides = [3072, 1, 96, 32]}, @DDR>)
                            outputs(%output_pad_alloc_1 : memref<1x32x32x2x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x2x!qElemType, #NHWC, @DDR>
    
    %concat_view_output = memref.alloc() : memref<1x32x32x6x!qElemType, #NHWC, @DDR>
    %out_subview_0 = VPUIP.SubView %concat_view_output [0, 0, 0, 0] [1, 32, 32, 1] : memref<1x32x32x6x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>
    %out_copy_0 = VPUIP.Copy inputs(%input_pad_copy_0 : memref<1x32x32x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_0 : memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>)
                            -> memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>
    %out_subview_1 = VPUIP.SubView %concat_view_output [0, 0, 0, 1] [1, 32, 32, 3] : memref<1x32x32x6x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>
    %input_copy = VPUIP.Copy inputs(%input_copy_to_ddr : memref<1x32x32x3x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_1 : memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>)
                            -> memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>
    %out_subview_2 = VPUIP.SubView %concat_view_output [0, 0, 0, 4] [1, 32, 32, 2] : memref<1x32x32x6x!qElemType, #NHWC, @DDR>
                                to memref<1x32x32x2x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>
    %out_copy_1 = VPUIP.Copy inputs(%input_pad_copy_1: memref<1x32x32x2x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_2 : memref<1x32x32x2x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>)
                            -> memref<1x32x32x2x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>
    
    %concat_view = VPUIP.ConcatView inputs(%out_copy_0, %input_copy, %out_copy_1 : 
                                    memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>, 
                                    memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>, 
                                    memref<1x32x32x2x!qElemType, {order = #NHWC, strides = [6144, 1, 192,  32]}, @DDR>) 
                                outputs(%concat_view_output : memref<1x32x32x6x!qElemType, #NHWC, @DDR>) -> memref<1x32x32x6x!qElemType, #NHWC, @DDR>
   
    return %concat_view: memref<1x32x32x6x!qElemType, #NHWC, @DDR>

    // CHECK:    VPUIP.ConcatView
    // CHECK-NOT:   memref<1x32x32x1x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, [@CMX_NN, 0]>, memref<1x32x32x3x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, [@CMX_NN, 0]>, memref<1x32x32x2x!qElemType, {order = #NHWC, strides = [34816, 1, 1088, 32]}, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT:%.+]] : memref<1x32x32x6x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x32x6x!qElemType, #NHWC, [@CMX_NN, 0]>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @dontMoveReflectPadToCmxDoesntFitInCmx
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1024x1024x3x!qElemType>)
func.func @dontMoveReflectPadToCmxDoesntFitInCmx(%arg0: memref<1x1024x1024x3x!qElemType>) -> memref<1x1024x1024x5x!qElemType, #NHWC, @DDR> {
    %alloc_cmx = memref.alloc() : memref<1x1024x1024x3x!qElemType, #NHWC, @CMX_NN>
    %copy_0 = VPUIP.Copy inputs (%arg0: memref<1x1024x1024x3x!qElemType>)
                        outputs (%alloc_cmx: memref<1x1024x1024x3x!qElemType, #NHWC, @CMX_NN>) -> memref<1x1024x1024x3x!qElemType, #NHWC, @CMX_NN>

    %alloc_ddr = memref.alloc() : memref<1x1024x1024x3x!qElemType, #NHWC, @DDR>
    %input_copy_to_ddr = VPUIP.Copy inputs (%copy_0: memref<1x1024x1024x3x!qElemType, #NHWC, @CMX_NN>)
                        outputs (%alloc_ddr: memref<1x1024x1024x3x!qElemType, #NHWC, @DDR>) -> memref<1x1024x1024x3x!qElemType, #NHWC, @DDR>
    
    %input_pad_0 = VPUIP.SubView %input_copy_to_ddr [0, 0, 0, 1] [1, 1024, 1024, 1] : memref<1x1024x1024x3x!qElemType, #NHWC, @DDR>
                                to memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [3145728, 1, 3072, 1024]}, @DDR>
    %output_pad_alloc_0 = memref.alloc() : memref<1x1024x1024x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_0 = VPUIP.Copy inputs(%input_pad_0 : memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [3145728, 1, 3072, 1024]}, @DDR>)
                            outputs(%output_pad_alloc_0 : memref<1x1024x1024x1x!qElemType, #NHWC, @DDR>) -> memref<1x1024x1024x1x!qElemType, #NHWC, @DDR>
                            
    %input_pad_1 = VPUIP.SubView %input_copy_to_ddr [0, 0, 0, 2] [1, 1024, 1024, 1] : memref<1x1024x1024x3x!qElemType, #NHWC, @DDR>
                                to memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [3145728, 1, 3072, 1024]}, @DDR>
    %output_pad_alloc_1 = memref.alloc() : memref<1x1024x1024x1x!qElemType, #NHWC, @DDR>
    %input_pad_copy_1 = VPUIP.Copy inputs(%input_pad_1 : memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [3145728, 1, 3072, 1024]}, @DDR>)
                            outputs(%output_pad_alloc_1 : memref<1x1024x1024x1x!qElemType, #NHWC, @DDR>) -> memref<1x1024x1024x1x!qElemType, #NHWC, @DDR>
    
    %concat_view_output = memref.alloc() : memref<1x1024x1024x5x!qElemType, #NHWC, @DDR>
    %out_subview_0 = VPUIP.SubView %concat_view_output [0, 0, 0, 0] [1, 1024, 1024, 1] : memref<1x1024x1024x5x!qElemType, #NHWC, @DDR>
                                to memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>
    %out_copy_0 = VPUIP.Copy inputs(%input_pad_copy_0 : memref<1x1024x1024x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_0 : memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>)
                            -> memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>
    %out_subview_1 = VPUIP.SubView %concat_view_output [0, 0, 0, 1] [1, 1024, 1024, 3] : memref<1x1024x1024x5x!qElemType, #NHWC, @DDR>
                                to memref<1x1024x1024x3x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>
    %input_copy = VPUIP.Copy inputs(%input_copy_to_ddr : memref<1x1024x1024x3x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_1 : memref<1x1024x1024x3x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>)
                            -> memref<1x1024x1024x3x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>
    %out_subview_2 = VPUIP.SubView %concat_view_output [0, 0, 0, 4] [1, 1024, 1024, 1] : memref<1x1024x1024x5x!qElemType, #NHWC, @DDR>
                                to memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>
    %out_copy_1 = VPUIP.Copy inputs(%input_pad_copy_1: memref<1x1024x1024x1x!qElemType, #NHWC, @DDR>)
                            outputs(%out_subview_2 : memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>)
                            -> memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>
    
    %concat_view = VPUIP.ConcatView inputs(%out_copy_0, %input_copy, %out_copy_1 : 
                                    memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>, 
                                    memref<1x1024x1024x3x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>, 
                                    memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, @DDR>) 
                                outputs(%concat_view_output : memref<1x1024x1024x5x!qElemType, #NHWC, @DDR>) -> memref<1x1024x1024x5x!qElemType, #NHWC, @DDR>
   
    return %concat_view: memref<1x1024x1024x5x!qElemType, #NHWC, @DDR>

    // CHECK:    VPUIP.ConcatView
    // CHECK-NOT:   memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, [@CMX_NN, 0]>, memref<1x1024x1024x3x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, [@CMX_NN, 0]>, memref<1x1024x1024x1x!qElemType, {order = #NHWC, strides = [5242880, 1, 5120, 1024]}, [@CMX_NN, 0]>) outputs([[CONCAT_VIEW_OUTPUT:%.+]] : memref<1x1024x1024x5x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x1024x1024x5x!qElemType, #NHWC, [@CMX_NN, 0]>
}
