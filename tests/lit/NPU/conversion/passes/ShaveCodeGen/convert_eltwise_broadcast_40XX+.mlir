//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" \
// RUN:     --convert-eltwise-layers-to-math | FileCheck %s
// REQUIRES: arch-NPU40XX

module @BroadcastMax {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x16xf16>
    DataInfo "input1" : tensor<1x1x1x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xf16>
  }

  func.func @main(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x1x1x16xf16>) -> tensor<1x1x16x16xf16> {
    %r = IE.Maximum(%arg0, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x16x16xf16>
    return %r : tensor<1x1x16x16xf16>
  }
// CHECK: #[[NCHW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
// CHECK: module @BroadcastMax
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x16x16xf16>, [[ARG1:%.+]]: tensor<1x1x1x16xf16>) -> tensor<1x1x16x16xf16> {
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[LINALG:%.+]] = linalg.generic {indexing_maps = [#[[NCHW]], #[[MAP_RHS]], #[[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]], [[ARG1]] : tensor<1x1x16x16xf16>, tensor<1x1x1x16xf16>) outs([[EMPTY]] : tensor<1x1x16x16xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:      [[MAX:%.+]] = arith.maximumf [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      linalg.yield [[MAX]] : f16
// CHECK-NEXT:    } -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    return [[LINALG]] : tensor<1x1x16x16xf16>
}

// -----

module @BroadcastDiv {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x16xf16>
    DataInfo "input1" : tensor<1x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xf16>
  }

  func.func @main(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x16xf16>) -> tensor<1x1x16x16xf16> {
    %r = IE.Divide(%arg0, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xf16>, tensor<1x16xf16> -> tensor<1x1x16x16xf16>
    return %r : tensor<1x1x16x16xf16>
  }
// CHECK: #[[NCHW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (0, d3)>
// CHECK: module @BroadcastDiv
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x16x16xf16>, [[ARG1:%.+]]: tensor<1x16xf16>) -> tensor<1x1x16x16xf16> {
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[LINALG:%.+]] = linalg.generic {indexing_maps = [#[[NCHW]], #[[MAP_RHS]], #[[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]], [[ARG1]] : tensor<1x1x16x16xf16>, tensor<1x16xf16>) outs([[EMPTY]] : tensor<1x1x16x16xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:      [[DIV:%.+]] = arith.divf [[LHS]], [[RHS]] fastmath<arcp> : f16
// CHECK-NEXT:      linalg.yield [[DIV]] : f16
// CHECK-NEXT:    } -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    return [[LINALG]] : tensor<1x1x16x16xf16>
}

// -----

module @BroadcastMin {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x16x16xf16>
    DataInfo "input1" : tensor<10x1x1x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<10x1x16x16xf16>
  }

  func.func @main(%arg0: tensor<1x16x16xf16>, %arg1: tensor<10x1x1x16xf16>) -> tensor<10x1x16x16xf16> {
    %r = IE.Minimum(%arg0, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x16xf16>, tensor<10x1x1x16xf16> -> tensor<10x1x16x16xf16>
    return %r : tensor<10x1x16x16xf16>
  }
// CHECK: #[[NCHW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
// CHECK: module @BroadcastMin
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x16x16xf16>, [[ARG1:%.+]]: tensor<10x1x1x16xf16>) -> tensor<10x1x16x16xf16> {
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<10x1x16x16xf16>
// CHECK-NEXT:    [[LINALG:%.+]] = linalg.generic {indexing_maps = [#[[MAP_LHS]], #[[MAP_RHS]], #[[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]], [[ARG1]] : tensor<1x16x16xf16>, tensor<10x1x1x16xf16>) outs([[EMPTY]] : tensor<10x1x16x16xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:      [[MIN:%.+]] = arith.minimumf [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      linalg.yield [[MIN]] : f16
// CHECK-NEXT:    } -> tensor<10x1x16x16xf16>
// CHECK-NEXT:    return [[LINALG]] : tensor<10x1x16x16xf16>
}
