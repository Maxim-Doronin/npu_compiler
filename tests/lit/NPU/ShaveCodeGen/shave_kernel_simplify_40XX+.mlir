//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --shave-kernel-simplify %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @RemoveInputPad {
  module @VPU.SW {
    func.func @PaddedInputSoftmax(%arg0: tensor<1x1x1x1008xf16>) -> tensor<1x1x1x1000xf16> {
      %cst = arith.constant 0.000000e+00 : f16
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1000xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1, 2, 3]] output_shape [1, 1, 1, 1000] : tensor<1000xf16> into tensor<1x1x1x1000xf16>
      %0 = tensor.empty() : tensor<1x1x1x1008xf16>
      %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<1x1x1x1008xf16>) -> tensor<1x1x1x1008xf16>
      %extracted_slice_0 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1000xf16>
      %expanded_1 = tensor.expand_shape %extracted_slice_0 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1000] : tensor<1000xf16> into tensor<1x1x1x1000xf16>
      %2 = linalg.softmax dimension(3) ins(%expanded : tensor<1x1x1x1000xf16>) outs(%expanded_1 : tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16>
      return %2 : tensor<1x1x1x1000xf16>

// CHECK:    func.func @PaddedInputSoftmax([[ARG0:%.+]]: tensor<1x1x1x1008xf16>)
// CHECK-DAG:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1000xf16>
// CHECK-DAG:      [[RESHAPED_SLICE:%.+]] = tensor.expand_shape [[SLICE]] {{.+}} : tensor<1000xf16> into tensor<1x1x1x1000xf16>
// CHECK-DAG:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-DAG:      [[OUT_INIT:%.+]] = linalg.fill ins({{%.+}} : f16) outs([[EMPTY]] : tensor<1x1x1x1000xf16>)
// CHECK-NEXT:     [[SM:%.+]] = linalg.softmax dimension(3) ins([[RESHAPED_SLICE]] : tensor<1x1x1x1000xf16>) outs([[OUT_INIT]] : tensor<1x1x1x1000xf16>)
// CHECK-NEXT:     return [[SM]] : tensor<1x1x1x1000xf16>
    }
  }
}
