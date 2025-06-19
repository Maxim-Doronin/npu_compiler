
//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// TODO host-compile options for lit test will be addressed with this ticket E#163523
// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --host-compile="enable-dynamic-shape-transformations=false scf-tiling=true scf-compute-ops-outlining=true use-memref-for-host-function-bufferization=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @CopyInputOutput
module @CopyInputOutput {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }

  // CHECK:  module @[[MODULE0:.+]] {
  // CHECK-NEXT:    func.func private @[[FUNC0:.+]](%[[ARG0:.+]]: memref<1x3x60x60xf16>, %[[ARG1:.+]]: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
  // CHECK:  func.func @main([[ARG0:%.+]]: memref<1x3x60x60xf16>, [[ARG1:%.+]]: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
  // CHECK-NEXT:    [[ALLOC:%.+]] = memref.alloc() : memref<1x3x60x60xf16>
  // CHECK-NEXT:    [[OUT0:%.+]] = Core.NestedCall @[[MODULE0]]::@[[FUNC0]]([[ARG0]], [[ALLOC]]) : (memref<1x3x60x60xf16>, memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
  // CHECK:  memref.copy [[OUT0]], [[ARG1]] : memref<1x3x60x60xf16> to memref<1x3x60x60xf16>
  // CHECK-NEXT:    return [[ARG1]] : memref<1x3x60x60xf16>

  func.func private @main_part1(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %0 = VPU.Copy(%arg0) : tensor<1x3x60x60xf16> -> tensor<1x3x60x60xf16>
    return %0 : tensor<1x3x60x60xf16>
  }

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %0 = call @main_part1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %0 : tensor<1x3x60x60xf16>

  }
}

// -----

// CHECK-LABEL: @StaticEltwiseNHWC
module @StaticEltwiseNHWC {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x720x1000xf16>
        DataInfo "input2" : tensor<1x16x720x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x720x1000xf16>
    }

    // CHECK:  module @[[MODULE0:.+]] {
    // CHECK-NEXT:    func.func private @[[FUNC0:.+]](%[[ARG0:.+]]: memref<1x16x90x1000xf16, #NHWC>, %[[ARG1:.+]]: memref<1x16x90x1000xf16, #NHWC>, %[[ARG2:.+]]: memref<1x16x90x1000xf16, #NHWC>) -> memref<1x16x90x1000xf16, #NHWC>
    func.func @main(%arg0: tensor<1x16x720x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>,
                    %arg1: tensor<1x16x720x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>)
          -> tensor<1x16x720x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {

    // CHECK:  func.func @main([[ARG0:%.+]]: memref<1x16x720x1000xf16, #NHWC>, [[ARG1:%.+]]: memref<1x16x720x1000xf16, #NHWC>, [[ARG2:%.+]]: memref<1x16x720x1000xf16, #NHWC>) -> memref<1x16x720x1000xf16, #NHWC>
        %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x720x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>,
            tensor<1x16x720x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                -> tensor<1x16x720x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        return %0 : tensor<1x16x720x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

        // TODO:    bufferization.to_tensor/to_memref will be removed once scf dialect bufferization is fixed
        // Track:   [E#168306]

        // CHECK:   [[ARG1_tensor:%.+]] = bufferization.to_tensor [[ARG1]] : memref<1x16x720x1000xf16, #NHWC>
        // CHECK:   [[ARG0_tensor:%.+]] = bufferization.to_tensor [[ARG0]] : memref<1x16x720x1000xf16, #NHWC>
        // CHECK:   [[OUTPUT_tensor:%.+]] = tensor.empty() : tensor<1x16x720x1000xf16, {order = #
        // CHECK:   [[RESULT_tensor:%.+]] = scf.for [[OFFSET:%.+]] = %c0 to %c720 step %c90 iter_args([[OUT_tensor_arg:%.+]] = [[OUTPUT_tensor]]) -> (tensor<1x16x720x1000xf16, {order = #NHWC}>)
        // CHECK:       [[slice_0_tensor:%.*]] = tensor.extract_slice [[ARG0_tensor]]
        // CHECK:       [[slice_0_memref:%.*]] = bufferization.to_memref [[slice_0_tensor]] : memref<1x16x90x1000xf16, #NHWC>
        // CHECK:       [[slice_1_tensor:%.*]] = tensor.extract_slice [[ARG1_tensor]]
        // CHECK:       [[slice_1_memref:%.*]] = bufferization.to_memref [[slice_1_tensor]] : memref<1x16x90x1000xf16, #NHWC>
        // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x16x90x1000xf16, #NHWC>
        // CHECK:       [[OUT0:%.+]] = Core.NestedCall @[[MODULE0]]::@[[FUNC0]]([[slice_0_memref]], [[slice_1_memref]], [[ALLOC]])
        // CHECK:       [[OUT0_tensor:%.+]] = bufferization.to_tensor [[OUT0]] : memref<1x16x90x1000xf16, #NHWC>
        // CHECK:       [[inserted_slice:%.+]] = tensor.insert_slice [[OUT0_tensor]] into [[OUT_tensor_arg]][0, 0, [[OFFSET]], 0] [1, 16, 90, 1000] [1, 1, 1, 1] : tensor<1x16x90x1000xf16, {order = #NHWC}> into tensor<1x16x720x1000
        // CHECK:       scf.yield [[inserted_slice]] : tensor<1x16x720x1000xf16, {order = #
        // CHECK:   [[RESULT_buffer:%.+]] = bufferization.to_memref [[RESULT_tensor]] : memref<1x16x720x1000xf16, #NHWC>
        // CHECK:   memref.copy [[RESULT_buffer]], [[ARG2]] : memref<1x16x720x1000xf16, #NHWC> to memref<1x16x720x1000xf16, #NHWC>
        // CHECK:   return [[ARG2]] : memref<1x16x720x1000xf16, #NHWC>
    }
}
