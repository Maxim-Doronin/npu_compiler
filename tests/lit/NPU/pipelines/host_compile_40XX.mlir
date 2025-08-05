//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
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

  func.func private @main_part1(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %0 = VPU.Copy(%arg0) : tensor<1x3x60x60xf16> -> tensor<1x3x60x60xf16>
    return %0 : tensor<1x3x60x60xf16>
  }

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %0 = call @main_part1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %0 : tensor<1x3x60x60xf16>
  }
  // CHECK:  module [[MODULE0:@.+]] attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<HostCompile>} {
  // CHECK:    func.func private [[FUNC0:@.+]]([[_:%.+]]: memref<1x3x60x60xf16, @DDR>, [[_:%.+]]: memref<1x3x60x60xf16, @DDR>)
  // CHECK-SAME:        -> memref<1x3x60x60xf16, @DDR> attributes {inliner_dispatch = #VPUIP.VPUIPInlinerDispatch} {
  // CHECK-COUNT-1: VPURT.Task  
  // CHECK-NOT: VPU.Copy

  // CHECK:  func.func @main([[ARG0:%.+]]: memref<1x3x60x60xf16>, [[ARG1:%.+]]: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
  // CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x3x60x60xf16>

  // CHECK:    [[TOKEN:%.+]], [[BODY_RESULTS:%.+]] = async.execute
  // CHECK:        [[OUT0:%.+]] = Core.NestedCall [[MODULE0]]::[[FUNC0]]([[ARG0]], [[ALLOC]])
  // CHECK:        async.yield [[OUT0]]
  // CHECK:    [[RESULT:%.+]] = async.await [[BODY_RESULTS]]

  // CHECK:    memref.copy [[RESULT]], [[ARG1]] : memref<1x3x60x60xf16> to memref<1x3x60x60xf16>
  // CHECK:    return [[ARG1]] : memref<1x3x60x60xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @StaticEltwiseNHWC
module @StaticEltwiseNHWC {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x720x1000xf16>
        DataInfo "input2" : tensor<1x16x720x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x720x1000xf16>
    }

    // CHECK:  module [[MODULE0:@.+]] attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<HostCompile>} {
    // CHECK:    func.func private [[FUNC0:@.+]]([[_:%.+]]: memref<1x90x1000x16xf16, @DDR>, [[_:%.+]]: memref<1x90x1000x16xf16, @DDR>, [[_:%.+]]: memref<1x90x1000x16xf16, @DDR>) -> memref<1x90x1000x16xf16, @DDR> attributes {inliner_dispatch = #VPUIP.VPUIPInlinerDispatch} {
    // CHECK-COUNT-25: VPURT.Task
    // CHECK-NOT: IE.Add
    func.func @main(%arg0: tensor<1x16x720x1000xf16, {order = #NHWC}>,
                    %arg1: tensor<1x16x720x1000xf16, {order = #NHWC}>)
          -> tensor<1x16x720x1000xf16, {order = #NHWC}> {
        %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x720x1000xf16, {order = #NHWC}>,
            tensor<1x16x720x1000xf16, {order = #NHWC}>
                -> tensor<1x16x720x1000xf16, {order = #NHWC}>
        return %0 : tensor<1x16x720x1000xf16, {order = #NHWC}>

        // CHECK:  func.func @main([[ARG0:%.+]]: memref<1x720x1000x16xf16>, [[ARG1:%.+]]: memref<1x720x1000x16xf16>, [[ARG2:%.+]]: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16>

        // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
        // CHECK-DAG: [[C720:%.+]] = arith.constant 720 : index
        // CHECK-DAG: [[C90:%.+]] = arith.constant 90 : index

        // CHECK: [[GROUP:%.+]] = async.create_group
        // CHECK: scf.for [[ARG3:%.+]] = [[C0]] to [[C720]] step [[C90]] {
        // CHECK:     [[SUBVIEW0:%.+]] = memref.subview [[ARG0]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
        // CHECK:     [[SUBVIEW1:%.+]] = memref.subview [[ARG1]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]

        // CHECK:     [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW0]]
        // CHECK:     [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW1]]

        // CHECK:     [[SUBVIEW2:%.+]] = memref.subview [[ARG2]][0, [[ARG3]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
        // CHECK:     [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW2]]
        // CHECK:     [[TOKEN:%.+]], [[BODY_RESULTS:%.+]] = async.execute
        // CHECK:         [[CALL_RES:%.+]] = Core.NestedCall [[MODULE0]]::[[FUNC0]]([[CAST0]], [[CAST1]], [[CAST2]])
        // CHECK:         async.yield [[CALL_RES]]
        // CHECK:     async.add_to_group [[TOKEN]], [[GROUP]]
        // CHECK:     [[RESULT:%.+]] = async.await [[BODY_RESULTS]]

        // CHECK: async.await_all [[GROUP]]

        // CHECK: return [[ARG2]] : memref<1x720x1000x16xf16>
    }
}
