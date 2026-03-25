//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-to-spatial-op --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertToSpatialInterpolation
module @ConvertToSpatialInterpolation {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x64xf16>
func.func @main(%arg0: tensor<1x16x16x64xf16>) -> tensor<1x32x32x64xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [1, 2],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
         tensor<1x16x16x64xf16> -> tensor<1x32x32x64xf16>

    return %0 : tensor<1x32x32x64xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NWCH} : tensor<1x16x16x64xf16> -> tensor<1x64x16x16xf16>
    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x64x32x32xf16> -> tensor<1x32x32x64xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x32x32x64xf16>
}
}

// -----

// CHECK-LABEL: @BypassSpatialInterpolation
module @BypassSpatialInterpolation {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x64x16x16xf16>
func.func @main(%arg0: tensor<1x64x16x16xf16>) -> tensor<1x64x32x32xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
         tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>

    return %0 : tensor<1x64x32x32xf16>

    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>

    // CHECK:       return [[INTERPOLATE]] : tensor<1x64x32x32xf16>
}
}

// -----

// CHECK-LABEL: @ConvertToSpatialInterpolationOnSingleDim
module @ConvertToSpatialInterpolationOnSingleDim {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK-LABEL: func.func @main
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x64xf16>
func.func @main(%arg0: tensor<1x16x16x64xf16>) -> tensor<1x32x16x64xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 32, 16, 64]} :
         tensor<1x16x16x64xf16> -> tensor<1x32x16x64xf16>

    return %0 : tensor<1x32x16x64xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NWCH} : tensor<1x16x16x64xf16> -> tensor<1x64x16x16xf16>
    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 64, 32, 16]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x16xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x64x32x16xf16> -> tensor<1x32x16x64xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x32x16x64xf16>
}
}

// -----

// CHECK-LABEL: @BypassSpatialInterpolationOnSingleDim
module @BypassSpatialInterpolationOnSingleDim {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x8x64x2xf16>
func.func @main(%arg0: tensor<1x8x64x2xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x8x64x2xf16> -> tensor<1x8x64x4xf16>


    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 8, 64, 4]} :
    // CHECK-SAME:                        tensor<1x8x64x2xf16> -> tensor<1x8x64x4xf16>

    // CHECK:       return [[INTERPOLATE]] : tensor<1x8x64x4xf16>
}
}

// -----

// CHECK-LABEL: @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SCALES
module @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SCALES {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16>
func.func @main(%arg0: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x4x32x4xf16> -> tensor<1x8x64x4xf16>

    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NWCH} : tensor<1x4x32x4xf16> -> tensor<1x4x4x32xf16>
    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                        sizes_attr = [1, 4, 8, 64]} : tensor<1x4x4x32xf16> -> tensor<1x4x8x64xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x4x8x64xf16> -> tensor<1x8x64x4xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x8x64x4xf16>
}
}

// -----

// CHECK-LABEL: @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SIZES
module @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SIZES {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16>
func.func @main(%arg0: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x4x32x4xf16> -> tensor<1x8x64x4xf16>

    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NWCH} : tensor<1x4x32x4xf16> -> tensor<1x4x4x32xf16>
    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
    // CHECK-SAME:                        sizes_attr = [1, 4, 8, 64]} : tensor<1x4x4x32xf16> -> tensor<1x4x8x64xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x4x8x64xf16> -> tensor<1x8x64x4xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x8x64x4xf16>
}
}

// -----

// CHECK-LABEL: @ConvertToSpatialRollAtChannelAndHeight
module @ConvertToSpatialRollAtChannelAndHeight {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x7x9x64xf16>)
func.func @main(%arg0: tensor<1x7x9x64xf16>) -> tensor<1x7x9x64xf16> {
    %shift = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[1, 2]> : tensor<2xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x7x9x64xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x7x9x64xf16>
    return %roll : tensor<1x7x9x64xf16>

    // CHECK-DAG:   [[AXES:%.+]] = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    // CHECK-DAG:   [[SHIFTS:%.+]] = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[INPUT_DATA]]) {order_value = #NWCH}
    // CHECK-SAME:                 tensor<1x7x9x64xf16> -> tensor<1x64x7x9xf16>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_TRANSPOSE]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x64x7x9xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x64x7x9xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[ROLL]]) {order_value = #NHWC}
    // CHECK-SAME:                 tensor<1x64x7x9xf16> -> tensor<1x7x9x64xf16>
    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x7x9x64xf16>
}
}

// -----

// CHECK-LABEL: @ConvertToSpatialRollAtChannel
module @ConvertToSpatialRollAtChannel {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x7x9x64xf16>)
func.func @main(%arg0: tensor<1x7x9x64xf16>) -> tensor<1x7x9x64xf16> {
    %shift = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x7x9x64xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x7x9x64xf16>
    return %roll : tensor<1x7x9x64xf16>

    // CHECK-DAG:   [[AXES:%.+]] = const.Declare tensor<1xsi32> = dense<2> : tensor<1xsi32>
    // CHECK-DAG:   [[SHIFTS:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[INPUT_DATA]]) {order_value = #NWCH}
    // CHECK-SAME:                 tensor<1x7x9x64xf16> -> tensor<1x64x7x9xf16>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_TRANSPOSE]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x64x7x9xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x64x7x9xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[ROLL]]) {order_value = #NHWC}
    // CHECK-SAME:                 tensor<1x64x7x9xf16> -> tensor<1x7x9x64xf16>
    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x7x9x64xf16>
}
}

// -----

// CHECK-LABEL: @NotConvertToSpatialRollDueToWidthNotAligned
module @NotConvertToSpatialRollDueToWidthNotAligned {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x64x9x10xf16>)
func.func @main(%arg0: tensor<1x64x9x10xf16>) -> tensor<1x64x9x10xf16> {
    %shift = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x64x9x10xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x64x9x10xf16>
    return %roll : tensor<1x64x9x10xf16>

    // CHECK-DAG:   [[AXES:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK-DAG:   [[SHIFTS:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_DATA]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x64x9x10xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x64x9x10xf16>
    // CHECK:       return [[ROLL]] : tensor<1x64x9x10xf16>
}
}

// -----

// CHECK-LABEL: @NotConvertToSpatialRollAtChannelAndWidth
module @NotConvertToSpatialRollAtChannelAndWidth {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x7x9x64xf16>)
func.func @main(%arg0: tensor<1x7x9x64xf16>) -> tensor<1x7x9x64xf16> {
    %shift = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[1, 3]> : tensor<2xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x7x9x64xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x7x9x64xf16>
    return %roll : tensor<1x7x9x64xf16>

    // CHECK-DAG:   [[AXES:%.+]] = const.Declare tensor<2xsi32> = dense<[1, 3]> : tensor<2xsi32>
    // CHECK-DAG:   [[SHIFTS:%.+]] = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_DATA]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x7x9x64xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x7x9x64xf16>
    // CHECK:       return [[ROLL]] : tensor<1x7x9x64xf16>
}
}
