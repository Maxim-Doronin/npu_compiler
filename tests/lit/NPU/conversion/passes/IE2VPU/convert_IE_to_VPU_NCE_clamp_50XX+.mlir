//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @Q_32To128_LRelu
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @Q_32To128_LRelu(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            clamp = {min = 3.200000e+01 : f64, max = 1.280000e+02 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 3.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.280000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e-01],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @F16_0To120_Tanh
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56xf16, {order = #NHWC}>)
func.func @F16_0To120_Tanh(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.Tanh<>,
            clamp = {min = 0.000000e+00 : f64, max = 1.200000e+02 : f64}
        } : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x256x56x56xf16, {order = #NHWC}>
            -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <TANH>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.200000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @F16_0To120_Intersect
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56xf16, {order = #NHWC}>)
func.func @F16_0To120_Intersect(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.Clamp<min = -5.000000e+00 : f64, max = 1.200000e+02 : f64>,
            clamp = {min = 0.000000e+00 : f64, max = 1.280000e+02 : f64}
        } : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x256x56x56xf16, {order = #NHWC}>
            -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LRELUX>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.200000e+02 : f64,
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:255>:f16, 0.5:127>

// CHECK-LABEL: @Q_Neg6To6_Exp
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @Q_Neg6To6_Exp(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.Exp<>,
            clamp = {min = -6.000000e+00 : f64, max = 6.000000e+00 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <EXP>,
    // CHECK-SAME:          clamp_low = -1.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.200000e+01 : f64,
    // CHECK-SAME:          prelu_alpha = [2.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 1.270000e+02 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @Q_0To128_LRelu
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @Q_0To128_LRelu(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                       -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            clamp = {min = 0.000000e+00 : f64, max = 1.280000e+02 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.280000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e-01],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @Q_32To128_NoOp
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @Q_32To128_NoOp(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            clamp = {min = 3.200000e+01 : f64, max = 1.280000e+02 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 3.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.280000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}
