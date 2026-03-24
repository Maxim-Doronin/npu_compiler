//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fold-unit-dim-reshapes %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @foo(%arg0: tensor<1x16x1x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<1x1x16x32xf32> {
  %caps = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x16x1x32xf32>, %arg1 as %arg3: tensor<16x32xf32>) {
    %4 = tensor.empty() : tensor<1x16x1x32xi8>
    %5 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg3 : tensor<1x16x1x32xf32>, tensor<16x32xf32>) outs(%4 : tensor<1x16x1x32xi8>) {
    ^bb0(%in: f32, %in_0: f32, %out: i8):
      %8 = arith.cmpf oeq, %in, %in_0 fastmath<nnan,nsz> : f32
      %9 = arith.extui %8 : i1 to i8
      linalg.yield %9 : i8
    } -> tensor<1x16x1x32xi8>
    %collapsed = tensor.collapse_shape %5 [[0, 1, 2, 3]] : tensor<1x16x1x32xi8> into tensor<512xi8>
    %expanded = tensor.expand_shape %collapsed [[0, 1, 2, 3]] output_shape [1, 1, 16, 32] : tensor<512xi8> into tensor<1x1x16x32xi8>
    %6 = tensor.empty() : tensor<1x1x16x32xf32>
    %7 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x1x16x32xi8>) outs(%6 : tensor<1x1x16x32xf32>) {
    ^bb0(%in: i8, %out: f32):
      %8 = arith.uitofp %in : i8 to f32
      linalg.yield %8 : f32
    } -> tensor<1x1x16x32xf32>
    IE.CGCYield %7 : tensor<1x1x16x32xf32>
  } -> tensor<1x1x16x32xf32>
  return %caps : tensor<1x1x16x32xf32>

// CHECK:  IE.CodeGenCapsule inputs({{%.*}} as {{%.+}}: tensor<1x16x1x32xf32>, {{%.+}} as {{%.+}}: tensor<16x32xf32>) {
// CHECK:      [[EQ:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) outs({{%.+}} : tensor<1x1x16x32xi8>) {
// CHECK-NEXT:      ^bb0([[IN0:%.+]]: f32, [[IN1:%.+]]: f32, {{%.+}}: i8):
// CHECK-NEXT:        [[CMP:%.+]] = arith.cmpf oeq, [[IN0]], [[IN1]] fastmath<nnan,nsz> : f32
// CHECK-NEXT:        [[EXT:%.+]] = arith.extui [[CMP]] : i1 to i8
// CHECK-NEXT:        linalg.yield [[EXT]] : i8
// CHECK-NEXT:      } -> tensor<1x1x16x32xi8>
// CHECK:      [[CONV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[EQ]] : tensor<1x1x16x32xi8>) outs({{%.+}} : tensor<1x1x16x32xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i8, {{%.+}}: f32):
// CHECK-NEXT:        [[TOFP:%.+]] = arith.uitofp [[IN]] : i8 to f32
// CHECK-NEXT:        linalg.yield [[TOFP]] : f32
// CHECK-NEXT:      } -> tensor<1x1x16x32xf32>
// CHECK-NEXT: IE.CGCYield [[CONV]] : tensor<1x1x16x32xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>

// CHECK-LABEL: @bar
func.func @bar(%arg0: tensor<1x34x3x4xf32>, %arg1: tensor<1x1x1x4xf32>) -> tensor<1x34x3x4xi8> {
  %2 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x34x3x4xf32>, %arg1 as %arg3: tensor<1x1x1x4xf32>) {
    %5 = tensor.empty() : tensor<1x34x3x4xi8>
    %6 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg3 : tensor<1x34x3x4xf32>, tensor<1x1x1x4xf32>) outs(%5 : tensor<1x34x3x4xi8>) {
      ^bb0(%in: f32, %in_0: f32, %out: i8):
        %7 = arith.cmpf oeq, %in, %in_0 fastmath<nnan,nsz> : f32
        %8 = arith.extui %7 : i1 to i8
        linalg.yield %8 : i8
      } -> tensor<1x34x3x4xi8>
      IE.CGCYield %6 : tensor<1x34x3x4xi8>
  } -> tensor<1x34x3x4xi8>
  return %2 : tensor<1x34x3x4xi8>

// CHECK:  [[EMPT:%.+]] = tensor.empty() : tensor<1x34x3x4xi8>
// CHECK:  [[OP:%.+]] = linalg.generic
// CHECK-SAME:     outs([[EMPT]] : tensor<1x34x3x4xi8>)
// CHECK:  IE.CGCYield [[OP]] : tensor<1x34x3x4xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @baz
func.func @baz(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf16> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1xf32>) {
    %6 = tensor.empty() : tensor<1x1x1x1xf16>
    %7 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x1x1x1xf32>) outs(%6 : tensor<1x1x1x1xf16>) {
      ^bb0(%in: f32, %out: f16):
        %8 = arith.truncf %in : f32 to f16
        linalg.yield %8 : f16
    } -> tensor<1x1x1x1xf16>
    IE.CGCYield %7 : tensor<1x1x1x1xf16>
  } -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

// CHECK:  [[EMPT:%.+]] = tensor.empty() : tensor<1x1x1x1xf16>
// CHECK:  [[OP:%.+]] = linalg.generic
// CHECK-SAME:     outs([[EMPT]] : tensor<1x1x1x1xf16>)
// CHECK:  IE.CGCYield [[OP]] : tensor<1x1x1x1xf16>

}
