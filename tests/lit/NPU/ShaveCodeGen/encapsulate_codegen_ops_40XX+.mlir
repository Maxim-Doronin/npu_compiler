//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --encapsulate-codegen-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @SingleCosLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res1 = IE.Cos(%cos_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res2 = IE.Cos(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res2 : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR2:%.+]] = IE.CodeGenCapsule inputs([[VAR1]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosWithExceptionLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res1 = IE.Cos(%cos_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    %sig = IE.Sigmoid(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16> // not supported by ShaveCodeGen, at least for now

    %cos_res2 = IE.Cos(%sig) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res2 : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // Unsupported layer should stay unwrapped
    // CHECK: [[SIG_RES:%.+]] = IE.Sigmoid([[VAR1]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR2:%.+]] = IE.CodeGenCapsule inputs([[SIG_RES]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @BinaryEltwiseLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "max" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %max_res = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %div_res = IE.Divide(%max_res, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %div_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[MAX_RES:%.+]] = IE.Maximum([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[MAX_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[DIV_RES:%.+]] = IE.Divide([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[DIV_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR1]] : tensor<1x1x1x1000xf16>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NWCH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @ChainWithReshapesAndSlice {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1008x1x1xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf32>
  }

  func.func @main(%arg0: tensor<1x1008x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1000xf32> {
    %0 = IE.Log(%arg0) : tensor<1x1008x1x1xf16, {order = #NHWC}> -> tensor<1x1008x1x1xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = #NHWC}> to tensor<1x1000x1x1xf16, {order = #NHWC}>
    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1000x1x1xf16, {order = #NHWC}> -> tensor<1x1000x1x1xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
    %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
    return %4 : tensor<1x1x1x1000xf32>

// CHECK:    [[LOGC:%.+]] = IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1008x1x1xf16, {order = [[NHWC]]}>) {
// CHECK:      [[LOG:%.+]] = IE.Log([[ARG1]]) : tensor<1x1008x1x1xf16, {order = [[NHWC]]}> -> tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:      IE.CGCYield [[LOG]] : tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:    } -> tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:    [[SLICEC:%.+]] = IE.CodeGenCapsule inputs([[LOGC]] as [[ARG1:%.+]]: tensor<1x1008x1x1xf16, {order = [[NHWC]]}>) {
// CHECK:      [[SLICE:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = [[NHWC]]}> to tensor<1x1000x1x1xf16, {order = [[NHWC]]}>
// CHECK:      IE.CGCYield [[SLICE]] : tensor<1x1000x1x1xf16, {order = [[NHWC]]}>
// CHECK:    } -> tensor<1x1000x1x1xf16, {order = [[NHWC]]}>
// CHECK:    [[PCC:%.+]] = IE.CodeGenCapsule inputs([[SLICEC]] as [[ARG1:%.+]]: tensor<1x1000x1x1xf16, {order = [[NHWC]]}>) {
// CHECK:      [[PC:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = [[NCHW]], mem_perm = [[NWCH]]} : tensor<1x1000x1x1xf16, {order = [[NHWC]]}> -> tensor<1x1000x1x1xf16>
// CHECK:      IE.CGCYield [[PC]] : tensor<1x1000x1x1xf16>
// CHECK:    } -> tensor<1x1000x1x1xf16>
// CHECK:    [[ARC:%.+]] = IE.CodeGenCapsule inputs([[PCC]] as [[ARG1:%.+]]: tensor<1x1000x1x1xf16>) {
// CHECK:      [[AR:%.+]] = IE.AffineReshape([[ARG1]]) {dim_mapping = {{\[\[}}0, 1, 2], [3], [3], [3{{\]\]}}, shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
// CHECK:      IE.CGCYield [[AR]] : tensor<1x1x1x1000xf16>
// CHECK:    } -> tensor<1x1x1x1000xf16>
// CHECK:    [[CONVC:%.+]] = IE.CodeGenCapsule inputs([[ARC]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK:      [[CONV:%.+]] = IE.Convert([[ARG1]]) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
// CHECK:      IE.CGCYield [[CONV]] : tensor<1x1x1x1000xf32>
// CHECK:    } -> tensor<1x1x1x1000xf32>
// CHECK:    return [[CONVC]] : tensor<1x1x1x1000xf32>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NWCH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @EdgeReshapesAndSlice {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1008xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1000x1x1xf32, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x1008x1x1xf16, {order = #NHWC}>) -> tensor<1x1000x1x1xf32, {order = #NHWC}> {
    %0 = IE.Log(%arg0) : tensor<1x1008x1x1xf16, {order = #NHWC}> -> tensor<1x1008x1x1xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = #NHWC}> to tensor<1x1000x1x1xf16, {order = #NHWC}>
    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1000x1x1xf16, {order = #NHWC}> -> tensor<1x1000x1x1xf16>
    %3 = IE.Sigmoid(%2) : tensor<1x1000x1x1xf16> -> tensor<1x1000x1x1xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
    %5 = IE.Convert(%4) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
    %6 = IE.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x1x1000xf32> -> tensor<1x1000x1x1xf32, {order = #NHWC}>
    return %6 : tensor<1x1000x1x1xf32, {order = #NHWC}>

// CHECK:    [[LOGC:%.+]] = IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1008x1x1xf16, {order = [[NHWC]]}>) {
// CHECK:      [[LOG:%.+]] = IE.Log([[ARG1]]) : tensor<1x1008x1x1xf16, {order = [[NHWC]]}> -> tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:      IE.CGCYield [[LOG]] : tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:    } -> tensor<1x1008x1x1xf16, {order = [[NHWC]]}>
// CHECK:    [[SLICE:%.+]] = IE.Slice [[LOGC]] [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = [[NHWC]]}> to tensor<1x1000x1x1xf16, {order = [[NHWC]]}>
// CHECK:    [[PC:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = [[NCHW]], mem_perm = [[NWCH]]} : tensor<1x1000x1x1xf16, {order = [[NHWC]]}> -> tensor<1x1000x1x1xf16>
// CHECK:    [[SIGM:%.+]] = IE.Sigmoid([[PC]]) : tensor<1x1000x1x1xf16> -> tensor<1x1000x1x1xf16>
// CHECK:    [[AR:%.+]] = IE.AffineReshape([[SIGM]]) {dim_mapping = {{\[\[}}0, 1, 2], [3], [3], [3{{\]\]}}, shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
// CHECK:    [[CONVC:%.+]] = IE.CodeGenCapsule inputs([[AR]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK:      [[CONV:%.+]] = IE.Convert([[ARG1:%.+]]) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
// CHECK:      IE.CGCYield [[CONV:%.+]] : tensor<1x1x1x1000xf32>
// CHECK:    } -> tensor<1x1x1x1000xf32>
// CHECK:    [[PC1:%.+]] = IE.PermuteCast([[CONVC]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x1x1x1000xf32> -> tensor<1x1000x1x1xf32, {order = [[NHWC]]}>
// CHECK:    return [[PC1]] : tensor<1x1000x1x1xf32, {order = [[NHWC]]}>
  }
}


// -----

module @RejectDynamicShape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<?x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<?x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<?x1x1x1000xf16>) -> tensor<?x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<?x1x1x1000xf16> -> tensor<?x1x1x1000xf16>
    return %cos_res : tensor<?x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<?x1x1x1000xf16>) -> tensor<?x1x1x1000xf16> {
    // CHECK-NOT: [[VAR0:%.+]] = IE.CodeGenCapsule 
  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0588235294117647:37>

module @AcceptPerTensorQuant {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x32x16x8xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x16x8xf16>
  }
 func.func @main(%arg0: tensor<1x32x16x8xf16>) -> tensor<1x32x16x8xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x32x16x8xf16> -> tensor<1x32x16x8x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x32x16x8x!qElemType> -> tensor<1x32x16x8xf16>
    return %1 : tensor<1x32x16x8xf16>

    // CHECK:    [[CAP1:%.+]] = IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x32x16x8xf16>) {
    // CHECK:      [[QUA:%.+]] = IE.Quantize([[ARG1]]) {dstElemType = !qElemType} : tensor<1x32x16x8xf16> -> tensor<1x32x16x8x!qElemType>
    // CHECK:      IE.CGCYield [[QUA]] : tensor<1x32x16x8x!qElemType>
    // CHECK:    } -> tensor<1x32x16x8x!qElemType>
    // CHECK:    [[CAP2:%.+]] = IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x32x16x8x!qElemType>) {
    // CHECK:      [[DEQ:%.+]] = IE.Dequantize([[ARG1]]) {dstElemType = f16} : tensor<1x32x16x8x!qElemType> -> tensor<1x32x16x8xf16>
    // CHECK:      IE.CGCYield [[DEQ]] : tensor<1x32x16x8xf16>
    // CHECK:    } -> tensor<1x32x16x8xf16>
  }
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.031372549019607843,0.030591299019607842,0.032935049019607844,0.029013480392156864,0.034497549019607845,0.027450980392156862,0.036090686274509801,0.025888480392156864}>

module @RejectPeAxisQuant {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x8x3x21xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x8x3x21xf16>
  }
 func.func @main(%arg0: tensor<1x8x3x21xf16>) -> tensor<1x8x3x21xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x8x3x21xf16> -> tensor<1x8x3x21x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x8x3x21x!qElemType> -> tensor<1x8x3x21xf16>
    return %1 : tensor<1x8x3x21xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x3x21xf16>) -> tensor<1x8x3x21xf16> 
    // CHECK-NOT: [[VAR0:%.+]] = IE.CodeGenCapsule 
  }
}

// -----

module @RejectBf16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x10x100xbf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x10x100xbf16>
  }

  func.func @main(%arg0: tensor<1x1x10x100xbf16>) -> tensor<1x1x10x100xbf16> {
    %cos_res = IE.Clamp(%arg0) {min = 1.0, max = 3.0} : tensor<1x1x10x100xbf16> -> tensor<1x1x10x100xbf16>
    return %cos_res : tensor<1x1x10x100xbf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x10x100xbf16>) -> tensor<1x1x10x100xbf16> 
    // CHECK-NOT: [[VAR0:%.+]] = IE.CodeGenCapsule 
  }
}
