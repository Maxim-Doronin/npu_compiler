//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger 8192 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --map-bilinear-interpolate-on-dpu %s | FileCheck %s
// REQUIRES: arch-NPU37XX


// CHECK-LABEL: @DoNotMapBilinearAlignCornersInterpolateOnDPUBecauseSmallChannel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x3x1024x1024xf16>
func.func @DoNotMapBilinearAlignCornersInterpolateOnDPUBecauseSmallChannel(%arg0: tensor<1x3x1024x1024xf16>) -> tensor<1x3x512x512xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <FLOOR>,
        antialias = false,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 0, 0],
        cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [5.000000e-01, 5.000000e-01],
        sizes_attr = [512, 512]} : tensor<1x3x1024x1024xf16> -> tensor<1x3x512x512xf16>
    return %0 : tensor<1x3x512x512xf16>

    // CHECK:   IE.Interpolate([[INPUT]])
}
