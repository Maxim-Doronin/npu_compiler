//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-aggregate-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

module @SoftmaxF32 {
  module @VPU.SW {
    func.func @SoftmaxF32(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
      %1 = tensor.empty() : tensor<1x16x4000x200xf32>
      %2 = linalg.softmax dimension(2) ins(%arg0 : tensor<1x16x4000x200xf32>) outs(%1 : tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32>
      return %2 : tensor<1x16x4000x200xf32>
    }
// CHECK: func.func @SoftmaxF32([[ARG1:%.+]]: tensor<1x16x4000x200xf32>)
// CHECK-NEXT:      [[OUT_EMPT:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
// CHECK-NEXT:      [[NEG_INF:%.+]] = arith.constant -3.40282347E+38 : f32
// CHECK-NEXT:      [[EMPT:%.+]] = tensor.empty() : tensor<1x16x200xf32>
// CHECK-NEXT:      [[MAX_OUT_INIT:%.+]] = linalg.fill ins([[NEG_INF]] : f32) outs([[EMPT]] : tensor<1x16x200xf32>) -> tensor<1x16x200xf32>
// CHECK-NEXT:      [[MAX:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG1]] : tensor<1x16x4000x200xf32>) outs([[MAX_OUT_INIT]] : tensor<1x16x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[SCALAR_MAX:%.+]] = arith.maximumf [[IN]], [[OUT]] fastmath<nnan,nsz> : f32
// CHECK-NEXT:        linalg.yield [[SCALAR_MAX]] : f32
// CHECK-NEXT:      } -> tensor<1x16x200xf32>
// CHECK-NEXT:      [[SUBEXP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG1]], [[MAX]] : tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>) outs([[OUT_EMPT]] : tensor<1x16x4000x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[IN1:%.+]]: f32, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[SUB:%.+]] = arith.subf [[IN]], [[IN1]] : f32
// CHECK-NEXT:        [[EXP:%.+]] = math.exp [[SUB]] fastmath<afn> : f32
// CHECK-NEXT:        linalg.yield [[EXP]] : f32
// CHECK-NEXT:      } -> tensor<1x16x4000x200xf32>
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[REDUCE_ADD_INIT:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<1x16x200xf32>) -> tensor<1x16x200xf32>
// CHECK-NEXT:      [[REDUCE_ADD:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[SUBEXP]] : tensor<1x16x4000x200xf32>) outs([[REDUCE_ADD_INIT]] : tensor<1x16x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[IN]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x16x200xf32>
// CHECK-NEXT:      [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[SUBEXP]], [[REDUCE_ADD]] : tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>) outs([[OUT_EMPT]] : tensor<1x16x4000x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[IN1]]: f32, {{.+}}: f32):
// CHECK-NEXT:        [[SCALAR_DIV:%.+]] = arith.divf [[IN]], [[IN1]] fastmath<arcp> : f32
// CHECK-NEXT:        linalg.yield [[SCALAR_DIV]] : f32
// CHECK-NEXT:      } -> tensor<1x16x4000x200xf32>
// CHECK-NEXT:      return [[DIV]] : tensor<1x16x4000x200xf32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module @SoftmaxF16 {
  module @VPU.SW {
    func.func @SoftmaxF16(%arg0: tensor<1x16x4000x200xf16>) -> tensor<1x16x4000x200xf16> {
    %1 = tensor.empty() : tensor<1x16x4000x200xf16>
    %2 = linalg.softmax dimension(3) ins(%arg0 : tensor<1x16x4000x200xf16>) outs(%1 : tensor<1x16x4000x200xf16>) -> tensor<1x16x4000x200xf16>
    return %2 : tensor<1x16x4000x200xf16>
  }
// CHECK: func.func @SoftmaxF16([[ARG1:%.+]]: tensor<1x16x4000x200xf16>) -> tensor<1x16x4000x200xf16>
// CHECK-NEXT:      [[OUT_EMPT:%.+]] = tensor.empty() : tensor<1x16x4000x200xf16>
// CHECK-NEXT:      [[NEG_INF:%.+]] = arith.constant -6.550400e+04 : f16
// CHECK-NEXT:      [[MAX_EMPT:%.+]] = tensor.empty() : tensor<1x16x4000xf16>
// CHECK-NEXT:      [[MAX_OUT_INIT:%.+]] = linalg.fill ins([[NEG_INF]] : f16) outs([[MAX_EMPT]] : tensor<1x16x4000xf16>) -> tensor<1x16x4000xf16>
// CHECK-NEXT:      [[MAX:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[ARG1]] : tensor<1x16x4000x200xf16>) outs([[MAX_OUT_INIT]] : tensor<1x16x4000xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SCALAR_MAX:%.+]] = arith.maximumf [[IN]], [[OUT]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:        linalg.yield [[SCALAR_MAX]] : f16
// CHECK-NEXT:      } -> tensor<1x16x4000xf16>
// CHECK-NEXT:      [[SUBEXP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG1]], [[MAX]] : tensor<1x16x4000x200xf16>, tensor<1x16x4000xf16>) outs([[OUT_EMPT]] : tensor<1x16x4000x200xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[IN1:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SUB:%.+]] = arith.subf [[IN]], [[IN1]] : f16
// CHECK-NEXT:        [[EXP:%.+]] = math.exp [[SUB]] fastmath<afn> : f16
// CHECK-NEXT:        linalg.yield [[EXP]] : f16
// CHECK-NEXT:      } -> tensor<1x16x4000x200xf16>
// CHECK-NEXT:      [[REDUCE_ADD_EMPT:%.+]] = tensor.empty() : tensor<1x16x4000xf32>
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[REDUCE_ADD_INIT:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[REDUCE_ADD_EMPT]] : tensor<1x16x4000xf32>) -> tensor<1x16x4000xf32>
// CHECK-NEXT:      [[REDUCE_ADD:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[SUBEXP]] : tensor<1x16x4000x200xf16>) outs([[REDUCE_ADD_INIT]] : tensor<1x16x4000xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[EXT]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x16x4000xf32>
// CHECK-NEXT:      [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[SUBEXP]], [[REDUCE_ADD]] : tensor<1x16x4000x200xf16>, tensor<1x16x4000xf32>) outs([[OUT_EMPT]] : tensor<1x16x4000x200xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[IN1:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[SCALAR_DIV:%.+]] = arith.divf [[EXT]], [[IN1]] fastmath<arcp> : f32
// CHECK-NEXT:        [[TRUNC:%.+]] = arith.truncf [[SCALAR_DIV]] : f32 to f16
// CHECK-NEXT:        linalg.yield [[TRUNC]] : f16
// CHECK-NEXT:      } -> tensor<1x16x4000x200xf16>
// CHECK-NEXT:      return [[DIV]] : tensor<1x16x4000x200xf16>
  }
}
