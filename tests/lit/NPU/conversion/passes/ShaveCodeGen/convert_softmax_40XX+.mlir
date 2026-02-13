//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NHWCToNHWCPaddedAxis1(
func.func @NHWCToNHWCPaddedAxis1(%arg0: tensor<1x16x4000x200xf32, {order=#NHWC}>) -> tensor<1x16x4000x200xf32, {order=#NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32, {order=#NHWC}>) {
     %1 = IE.SoftMax(%arg1) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x16x4000x200xf32, {order=#NHWC}> -> tensor<1x16x4000x200xf32, {order=#NHWC}>
    IE.CGCYield %1 : tensor<1x16x4000x200xf32, {order=#NHWC}>
  } -> tensor<1x16x4000x200xf32, {order=#NHWC}>
  return %0 : tensor<1x16x4000x200xf32, {order=#NHWC}>

// CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xf32>) {
// CHECK-NEXT:      [[IN_UNPAD:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x16xf32> to tensor<1x4000x200x12xf32>
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPT:%.+]] = tensor.empty() : tensor<1x4000x200x16xf32>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<1x4000x200x16xf32>) -> tensor<1x4000x200x16xf32>
// CHECK-NEXT:      [[OUT_SLICE:%.+]] = tensor.extract_slice [[FILL]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x16xf32> to tensor<1x4000x200x12xf32>
// CHECK-NEXT:      [[SM:%.+]] = linalg.softmax dimension(3) ins([[IN_UNPAD]] : tensor<1x4000x200x12xf32>) outs([[OUT_SLICE]] : tensor<1x4000x200x12xf32>) -> tensor<1x4000x200x12xf32>
// CHECK-NEXT:      [[PADDED:%.+]] = tensor.insert_slice [[SM]] into [[FILL]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x12xf32> into tensor<1x4000x200x16xf32>
// CHECK-NEXT:      IE.CGCYield [[PADDED]] : tensor<1x4000x200x16xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK: func.func @NHWCToNWCHPaddedAxis1(
func.func @NHWCToNWCHPaddedAxis1(%arg0: tensor<1x16x4000x200xf32, {order=#NHWC}>) -> tensor<1x16x4000x200xf32, {order=#NWCH}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32, {order=#NHWC}>) {
     %1 = IE.SoftMax(%arg1) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x16x4000x200xf32, {order=#NHWC}> -> tensor<1x16x4000x200xf32, {order=#NWCH}>
    IE.CGCYield %1 : tensor<1x16x4000x200xf32, {order=#NWCH}>
  } -> tensor<1x16x4000x200xf32, {order=#NWCH}>
  return %0 : tensor<1x16x4000x200xf32, {order=#NWCH}>

// CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1]]: tensor<1x4000x200x16xf32>) {
// CHECK-NEXT:      [[IN_UNPAD:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x16xf32> to tensor<1x4000x200x12xf32>
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPT:%.+]] = tensor.empty() : tensor<1x200x4000x16xf32>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<1x200x4000x16xf32>) -> tensor<1x200x4000x16xf32>
// CHECK-NEXT:      [[OUT_SLICE:%.+]] = tensor.extract_slice [[FILL]][0, 0, 0, 0] [1, 200, 4000, 12] [1, 1, 1, 1] : tensor<1x200x4000x16xf32> to tensor<1x200x4000x12xf32>
// CHECK-NEXT:      [[SM_OUT:%.+]] = tensor.empty() : tensor<1x4000x200x12xf32>
// CHECK-NEXT:      [[SM:%.+]] = linalg.softmax dimension(3) ins([[IN_UNPAD]] : tensor<1x4000x200x12xf32>) outs([[SM_OUT]] : tensor<1x4000x200x12xf32>) -> tensor<1x4000x200x12xf32>
// CHECK-NEXT:      [[SM_TRANSPOSED:%.+]] = linalg.transpose ins([[SM]] : tensor<1x4000x200x12xf32>) outs([[OUT_SLICE]] : tensor<1x200x4000x12xf32>) permutation = [0, 2, 1, 3]
// CHECK-NEXT:      [[PADDED:%.+]] = tensor.insert_slice [[SM_TRANSPOSED]] into [[FILL]][0, 0, 0, 0] [1, 200, 4000, 12] [1, 1, 1, 1] : tensor<1x200x4000x12xf32> into tensor<1x200x4000x16xf32>
// CHECK-NEXT:      IE.CGCYield [[PADDED]] : tensor<1x200x4000x16xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WHCN = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>

// CHECK: func.func @NHWCToNWCHPaddedAxis3(
func.func @NHWCToNWCHPaddedAxis3(%arg0: tensor<1x16x4000x200xf32, {order=#NHWC}>) -> tensor<1x16x4000x200xf32, {order=#WHCN}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32, {order=#NHWC}>) {
     %1 = IE.SoftMax(%arg1) {axisInd = 3 : i64, padSize = 4 : i64} : tensor<1x16x4000x200xf32, {order=#NHWC}> -> tensor<1x16x4000x200xf32, {order=#WHCN}>
    IE.CGCYield %1 : tensor<1x16x4000x200xf32, {order=#WHCN}>
  } -> tensor<1x16x4000x200xf32, {order=#WHCN}>
  return %0 : tensor<1x16x4000x200xf32, {order=#WHCN}>

// CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xf32>) {
// CHECK-NEXT:      [[IN_UNPAD:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 4000, 196, 16] [1, 1, 1, 1] : tensor<1x4000x200x16xf32> to tensor<1x4000x196x16xf32>
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPT:%.+]] = tensor.empty() : tensor<200x4000x16x1xf32>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<200x4000x16x1xf32>) -> tensor<200x4000x16x1xf32>
// CHECK-NEXT:      [[OUT_SLICE:%.+]] = tensor.extract_slice [[FILL]][0, 0, 0, 0] [196, 4000, 16, 1] [1, 1, 1, 1] : tensor<200x4000x16x1xf32> to tensor<196x4000x16x1xf32>
// CHECK-NEXT:      [[SM_OUT:%.+]] = tensor.empty() : tensor<1x4000x196x16xf32>
// CHECK-NEXT:      [[SM:%.+]] = linalg.softmax dimension(2) ins([[IN_UNPAD]] : tensor<1x4000x196x16xf32>) outs([[SM_OUT]] : tensor<1x4000x196x16xf32>) -> tensor<1x4000x196x16xf32>
// CHECK-NEXT:      [[SM_TRANSPOSED:%.+]] = linalg.transpose ins([[SM]] : tensor<1x4000x196x16xf32>) outs([[OUT_SLICE]] : tensor<196x4000x16x1xf32>) permutation = [2, 1, 3, 0]
// CHECK-NEXT:      [[PADDED:%.+]] = tensor.insert_slice [[SM_TRANSPOSED]] into [[FILL]][0, 0, 0, 0] [196, 4000, 16, 1] [1, 1, 1, 1] : tensor<196x4000x16x1xf32> into tensor<200x4000x16x1xf32>
// CHECK-NEXT:      IE.CGCYield [[PADDED]] : tensor<200x4000x16x1xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WHCN = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>

func.func @NHWCToNWCHAxis1(%arg0: tensor<1x16x4000x200xf32, {order=#NHWC}>) -> tensor<1x16x4000x200xf32, {order=#WHCN}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32, {order=#NHWC}>) {
     %1 = IE.SoftMax(%arg1) {axisInd = 1 : i64} : tensor<1x16x4000x200xf32, {order=#NHWC}> -> tensor<1x16x4000x200xf32, {order=#WHCN}>
    IE.CGCYield %1 : tensor<1x16x4000x200xf32, {order=#WHCN}>
  } -> tensor<1x16x4000x200xf32, {order=#WHCN}>
  return %0 : tensor<1x16x4000x200xf32, {order=#WHCN}>

// CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xf32>) {
// CHECK-NEXT:      [[SM_OUT:%.+]] = tensor.empty() : tensor<1x4000x200x16xf32>
// CHECK-NEXT:      [[SM:%.+]] = linalg.softmax dimension(3) ins([[ARG1]] : tensor<1x4000x200x16xf32>) outs([[SM_OUT]] : tensor<1x4000x200x16xf32>) -> tensor<1x4000x200x16xf32>
// CHECK-NEXT:      [[OUT:%.+]] = tensor.empty() : tensor<200x4000x16x1xf32>
// CHECK-NEXT:      [[SM_TRANSPOSED:%.+]] = linalg.transpose ins([[SM]] : tensor<1x4000x200x16xf32>) outs([[OUT]] : tensor<200x4000x16x1xf32>) permutation = [2, 1, 3, 0]
// CHECK-NEXT:      IE.CGCYield [[SM_TRANSPOSED]] : tensor<200x4000x16x1xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NHWCToNHWCAxis1NullPad(%arg0: tensor<1x16x4000x200xf32, {order=#NHWC}>) -> tensor<1x16x4000x200xf32, {order=#NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32, {order=#NHWC}>) {
     %1 = IE.SoftMax(%arg1) {axisInd = 1 : i64, padSize = 0 : i64} : tensor<1x16x4000x200xf32, {order=#NHWC}> -> tensor<1x16x4000x200xf32, {order=#NHWC}>
    IE.CGCYield %1 : tensor<1x16x4000x200xf32, {order=#NHWC}>
  } -> tensor<1x16x4000x200xf32, {order=#NHWC}>
  return %0 : tensor<1x16x4000x200xf32, {order=#NHWC}>

// CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xf32>) {
// CHECK-NEXT:      [[SM_OUT:%.+]] = tensor.empty() : tensor<1x4000x200x16xf32>
// CHECK-NEXT:      [[SM:%.+]] = linalg.softmax dimension(3) ins([[ARG1]] : tensor<1x4000x200x16xf32>) outs([[SM_OUT]] : tensor<1x4000x200x16xf32>) -> tensor<1x4000x200x16xf32>
// CHECK-NEXT:      IE.CGCYield [[SM]] : tensor<1x4000x200x16xf32>
}
