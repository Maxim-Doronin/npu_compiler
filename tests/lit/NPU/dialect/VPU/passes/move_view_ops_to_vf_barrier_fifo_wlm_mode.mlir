//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --move-view-ops-to-vf="workload-management-mode=PWLM_V1_BARRIER_FIFO" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveShapeCast
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x48x640x40xf16, {order = #NHWC}>
func.func @MoveShapeCast(%arg0: tensor<1x48x640x40xf16, {order = #NHWC}>) -> tensor<1x32x320x320xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x48x3x2xf16, {order = #NHWC}> = dense<1.0> : tensor<256x48x3x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x48x640x40xf16, {order = #NHWC}>,
                             %cst as %arg2: tensor<256x48x3x2xf16, {order = #NHWC}>,
                             %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x256x320x40xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                                          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                          pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
                                                          ppe = #VPU.PPEInt<mode = <NOOP>,
                                                          clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                                          rawFilterShape = [256, 48, 3, 2], strides = [2, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<256x48x3x2xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x320x40xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.ShapeCast {shape = [1, 32, 320, 320]} inputs(%0 : tensor<1x256x320x40xf16, {order = #NHWC}>) -> tensor<1x32x320x320xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x32x320x320xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x320x320xf16, {order = #NHWC}> {
      %inner = VPU.Swish(%arg1) {beta_value = 1.000000e+00 : f64,
                                 multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x320x320xf16, {order = #NHWC}> -> tensor<1x32x320x320xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %2 : tensor<1x32x320x320xf16, {order = #NHWC}>


    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]]
    //CHECK:  [[SHAPECAST:%.+]] = VPU.ShapeCast
    //CHECK:  [[SWISH:%.+]] = VPU.Swish([[SHAPECAST]])
    //CHECK:  VPU.Yield [[SWISH]]
    //CHECK:  return [[VF1]] : tensor<1x32x320x320xf16, {order = #NHWC}>
}
