//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-reduce-ops-with-mem-permute %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InsertMemPermuteBeforeAndAfterReduceSum
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @InsertMemPermuteBeforeAndAfterReduceSum(%arg0: tensor<1x1x256x768xf32>) -> tensor<1x1x1x768xf32> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x1x768xf32>

  return %0 : tensor<1x1x1x768xf32>

  // CHECK:        [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x256x768xf32> -> tensor<1x1x256x768xf32, {order = #NCWH}>
  // CHECK:        [[REDUCESUM:%.+]] = IE.ReduceSum([[MEMPERMUTE]]) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32, {order = #NCWH}> -> tensor<1x1x1x768xf32, {order = #NCWH}>
  // CHECK:        [[MEMPERMUTE_RESULT:%.+]] = IE.MemPermute([[REDUCESUM]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x1x768xf32, {order = #NCWH}> -> tensor<1x1x1x768xf32>

  // CHECK:         return [[MEMPERMUTE_RESULT]] : tensor<1x1x1x768xf32>
}

// -----

// CHECK-LABEL: @NotInsertMemPermuteBeforeAndAfterReduceSumAsInnerMostDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @NotInsertMemPermuteBeforeAndAfterReduceSumAsInnerMostDim(%arg0: tensor<1x1x256x768xf32>) -> tensor<1x1x256x1xf32> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x256x1xf32>

  return %0 : tensor<1x1x256x1xf32>

  // CHECK:        [[REDUCESUM:%.+]] = IE.ReduceSum([[INPUT]]) {axes_value = [3], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x256x1xf32>

  // CHECK:         return [[REDUCESUM]] : tensor<1x1x256x1xf32>
}

// -----

// CHECK-LABEL: @NotInsertMemPermuteBeforeAndAfterReduceSumAsNonOneDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x256x768xf32>
func.func @NotInsertMemPermuteBeforeAndAfterReduceSumAsNonOneDim(%arg0: tensor<1x16x256x768xf32>) -> tensor<1x1x256x1xf32> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x16x256x768xf32> -> tensor<1x1x256x1xf32>

  return %0 : tensor<1x1x256x1xf32>

  // CHECK:        [[REDUCESUM:%.+]] = IE.ReduceSum([[INPUT]]) {axes_value = [1, 3], keep_dims} : tensor<1x16x256x768xf32> -> tensor<1x1x256x1xf32>

  // CHECK:         return [[REDUCESUM]] : tensor<1x1x256x1xf32>
}

// -----

// CHECK-LABEL: @NotInsertMemPermuteBeforeAndAfterReduceSumAsMultiUses
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @NotInsertMemPermuteBeforeAndAfterReduceSumAsMultiUses(%arg0: tensor<1x1x256x768xf32>) -> (tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>) {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  %0 = IE.ReduceSum(%arg0) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x1x768xf32>
  %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x768xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x768xf32>

  return %0, %1 : tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>

  // CHECK-DAG:    [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK:        [[REDUCESUM:%.+]] = IE.ReduceSum([[INPUT]]) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x1x768xf32>
  // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[REDUCESUM]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x768xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x768xf32>

  // CHECK:         return [[REDUCESUM]], [[MULTIPLY]] : tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InsertMemPermuteBeforeAndAfterReduceMean
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @InsertMemPermuteBeforeAndAfterReduceMean(%arg0: tensor<1x1x256x768xf32>) -> tensor<1x1x1x768xf32> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x1x768xf32>

  return %0 : tensor<1x1x1x768xf32>

  // CHECK:        [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x256x768xf32> -> tensor<1x1x256x768xf32, {order = #NCWH}>
  // CHECK:        [[REDUCEMEAN:%.+]] = IE.ReduceMean([[MEMPERMUTE]]) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32, {order = #NCWH}> -> tensor<1x1x1x768xf32, {order = #NCWH}>
  // CHECK:        [[MEMPERMUTE_RESULT:%.+]] = IE.MemPermute([[REDUCEMEAN]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x1x768xf32, {order = #NCWH}> -> tensor<1x1x1x768xf32>

  // CHECK:         return [[MEMPERMUTE_RESULT]] : tensor<1x1x1x768xf32>
}

// -----

// CHECK-LABEL: @NotInsertMemPermuteBeforeAndAfterReduceMeanAsInnerMostDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @NotInsertMemPermuteBeforeAndAfterReduceMeanAsInnerMostDim(%arg0: tensor<1x1x256x768xf32>) -> tensor<1x1x256x1xf32> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x256x1xf32>

  return %0 : tensor<1x1x256x1xf32>

  // CHECK:        [[REDUCEMEAN:%.+]] = IE.ReduceMean([[INPUT]]) {axes_value = [3], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x256x1xf32>

  // CHECK:         return [[REDUCEMEAN]] : tensor<1x1x256x1xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InsertMemPermuteBeforeAndAfterReduceMin
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @InsertMemPermuteBeforeAndAfterReduceMin(%arg0: tensor<1x1x256x768xf32>) -> tensor<1x1x1x768xf32> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x1x768xf32>

  return %0 : tensor<1x1x1x768xf32>

  // CHECK:        [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x256x768xf32> -> tensor<1x1x256x768xf32, {order = #NCWH}>
  // CHECK:        [[REDUCEMIN:%.+]] = IE.ReduceMin([[MEMPERMUTE]]) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32, {order = #NCWH}> -> tensor<1x1x1x768xf32, {order = #NCWH}>
  // CHECK:        [[MEMPERMUTE_RESULT:%.+]] = IE.MemPermute([[REDUCEMIN]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x1x768xf32, {order = #NCWH}> -> tensor<1x1x1x768xf32>

  // CHECK:         return [[MEMPERMUTE_RESULT]] : tensor<1x1x1x768xf32>
}

// -----

// CHECK-LABEL: @NotInsertMemPermuteBeforeAndAfterReduceMinAsInnerMostDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @NotInsertMemPermuteBeforeAndAfterReduceMinAsInnerMostDim(%arg0: tensor<1x1x256x768xf32>) -> tensor<1x1x256x1xf32> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x256x1xf32>

  return %0 : tensor<1x1x256x1xf32>

  // CHECK:        [[REDUCEMIN:%.+]] = IE.ReduceMin([[INPUT]]) {axes_value = [3], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x256x1xf32>

  // CHECK:         return [[REDUCEMIN]] : tensor<1x1x256x1xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InsertMemPermuteBeforeAndAfterReduceMax
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @InsertMemPermuteBeforeAndAfterReduceMax(%arg0: tensor<1x1x256x768xf32>) -> tensor<1x1x1x768xf32> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x1x768xf32>

  return %0 : tensor<1x1x1x768xf32>

  // CHECK:        [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x256x768xf32> -> tensor<1x1x256x768xf32, {order = #NCWH}>
  // CHECK:        [[REDUCEMAX:%.+]] = IE.ReduceMax([[MEMPERMUTE]]) {axes_value = [2], keep_dims} : tensor<1x1x256x768xf32, {order = #NCWH}> -> tensor<1x1x1x768xf32, {order = #NCWH}>
  // CHECK:        [[MEMPERMUTE_RESULT:%.+]] = IE.MemPermute([[REDUCEMAX]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x1x768xf32, {order = #NCWH}> -> tensor<1x1x1x768xf32>

  // CHECK:         return [[MEMPERMUTE_RESULT]] : tensor<1x1x1x768xf32>
}

// -----

// CHECK-LABEL: @NotInsertMemPermuteBeforeAndAfterReduceMaxAsInnerMostDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x768xf32>
func.func @NotInsertMemPermuteBeforeAndAfterReduceMaxAsInnerMostDim(%arg0: tensor<1x1x256x768xf32>) -> tensor<1x1x256x1xf32> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x256x1xf32>

  return %0 : tensor<1x1x256x1xf32>

  // CHECK:        [[REDUCEMAX:%.+]] = IE.ReduceMax([[INPUT]]) {axes_value = [3], keep_dims} : tensor<1x1x256x768xf32> -> tensor<1x1x256x1xf32>

  // CHECK:         return [[REDUCEMAX]] : tensor<1x1x256x1xf32>
}
