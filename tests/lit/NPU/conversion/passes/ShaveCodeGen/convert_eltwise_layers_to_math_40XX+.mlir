//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleCosLayer
module @SingleCosLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
    // CHECK-NEXT:      [[COS:%.+]] = math.cos [[EXT]] : f32
    // CHECK-NEXT:      [[TRUNC:%.+]] = arith.truncf [[COS]] : f32 to f16
    // CHECK-NEXT:      linalg.yield [[TRUNC]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    return [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Divide

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleDivFPLayer
module @SingleDivFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %res = IE.Divide(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }: tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %res : tensor<1x1x1x1000xf16>

// CHECK:  func.func @main([[ARG0:%.*]]: tensor<1x1x1x1000xf16>, [[ARG1:%.*]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]], [[ARG1]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[OP:%.+]] = arith.divf [[LHS]], [[RHS]] fastmath<arcp> : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    return [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleDivSILayer
module @SingleDivSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %res = IE.Divide(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }: tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
    return %res : tensor<1x1x1x1000xsi32>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x1000xsi32>, [[RHS:%.+]]: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[LHS_BC]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.divsi [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    return [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleDivUILayer
module @SingleDivUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %res = IE.Divide(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }: tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
    return %res : tensor<1x1x1x1000xui32>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x1000xui32>, [[RHS:%.+]]: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[LHS_BC]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.divui [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    return [[RET]] : tensor<1x1x1x1000xui32>
  }
}

// -----
// IE.Maximum

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMaxFPLayer
module @SingleMaxFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %res = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }: tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %res : tensor<1x1x1x1000xf16>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x1000xf16>, [[RHS:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[OP:%.+]] = arith.maximumf [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    return [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMaxSILayer
module @SingleMaxSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %res = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }: tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
    return %res : tensor<1x1x1x1000xsi32>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x1000xsi32>, [[RHS:%.+]]: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[LHS_BC]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.maxsi [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    return [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMaxUILayer
module @SingleMaxUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %res = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }: tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
    return %res : tensor<1x1x1x1000xui32>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x1000xui32>, [[RHS:%.+]]: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[LHS_BC]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.maxui [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    return [[RET]] : tensor<1x1x1x1000xui32>
  }
}

// -----
// IE.Minimum

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMinFPLayer
module @SingleMinFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %res = IE.Minimum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }: tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %res : tensor<1x1x1x1000xf16>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x1000xf16>, [[RHS:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS]], [[RHS]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[LHS]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[OP:%.+]] = arith.minimumf [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    return [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMinSILayer
module @SingleMinSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %res = IE.Minimum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }: tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
    return %res : tensor<1x1x1x1000xsi32>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x1000xsi32>, [[RHS:%.+]]: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[LHS_BC]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.minsi [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    return [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMinUILayer
module @SingleMinUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %res = IE.Minimum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> }: tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
    return %res : tensor<1x1x1x1000xui32>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x1000xui32>, [[RHS:%.+]]: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[LHS_BC]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.minui [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    return [[RET]] : tensor<1x1x1x1000xui32>
  }
}

// -----
// Dynamic input shape

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMinUILayerDynamic
module @SingleMinUILayerDynamic {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x?xui32>
    DataInfo "input1" : tensor<1x1x1x?xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x?xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x?xui32>, %arg1: tensor<1x1x1x?xui32>) -> tensor<1x1x1x?xui32> {
    %res = IE.Minimum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> }: tensor<1x1x1x?xui32>, tensor<1x1x1x?xui32> -> tensor<1x1x1x?xui32>
    return %res : tensor<1x1x1x?xui32>

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x1x?xui32>, [[RHS:%.+]]: tensor<1x1x1x?xui32>) -> tensor<1x1x1x?xui32> {
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS]] : tensor<1x1x1x?xui32> to tensor<1x1x1x?xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS]] : tensor<1x1x1x?xui32> to tensor<1x1x1x?xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x?xi32>, tensor<1x1x1x?xi32>) outs([[LHS_BC]] : tensor<1x1x1x?xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.minui [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x?xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x?xi32> to tensor<1x1x1x?xui32>
// CHECK-NEXT:    return [[RET]] : tensor<1x1x1x?xui32>
  }
}

// -----

module @SingleLogLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "log" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %log_res = IE.Log(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %log_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = math.log [[ARG0]] fastmath<afn> : tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
	}
}
// -----

module @SingleExpLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "exp" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %exp_res = IE.Exp(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %exp_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = math.exp [[ARG0]] fastmath<afn> : tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
	}
}

// -----
// IE.Sin

module @SingleSinLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "sin" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %res = IE.Sin(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
    // CHECK-NEXT:      [[SIN:%.+]] = math.sin [[EXT]] : f32
    // CHECK-NEXT:      [[TRUNC:%.+]] = arith.truncf [[SIN]] : f32 to f16
    // CHECK-NEXT:      linalg.yield [[TRUNC]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    return [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Sqrt

module @SingleSqrtLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "sqrt" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %Sqrt_res = IE.Sqrt(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %Sqrt_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = math.sqrt [[ARG0]] fastmath<afn> : tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
    // CHECK: }
  }
}

// -----
// IE.Round

module @SingleRoundLayerHalfToEven  {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "Round" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %Round_res = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %Round_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[VAR0:%.+]] = math.roundeven [[IN]] : f16
    // CHECK-NEXT:      linalg.yield [[VAR0]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    return [[LINALG_OP]] : tensor<1x1x1x1000xf16>
    }
  }

// -----

module @SingleRoundLayerHalfAwayFromZero {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "Round" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %Round_res = IE.Round(%arg0) {mode = #IE.round_mode<HALF_AWAY_FROM_ZERO>} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %Round_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[VAR0:%.+]] = math.round [[IN]] : f16
    // CHECK-NEXT:      linalg.yield [[VAR0]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    return [[LINALG_OP]] : tensor<1x1x1x1000xf16>
    }
  }

// -----
// IE.Erf

module @SingleErfLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "Erf" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %Erf_res = IE.Erf(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %Erf_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
    // CHECK-NEXT:      [[ERF:%.+]] = math.erf [[EXT]] : f32
    // CHECK-NEXT:      [[TRUNC:%.+]] = arith.truncf [[ERF]] : f32 to f16
    // CHECK-NEXT:      linalg.yield [[TRUNC]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    return [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Convert

module @SingleConvertFPToSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xsi32> {
    %res = IE.Convert(%arg0) {dstElemType = si32}: tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xsi32>
    return %res : tensor<1x1x1x1000xsi32>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xsi32> {
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK: ^bb0([[ARG0:%.+]]: f16, {{%.+}}: i32):
// CHECK: [[OP:%.+]] = arith.fptosi %{{.+}} : f16 to i32
// CHECK: linalg.yield [[OP]] : i32
// CHECK: [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK: return %[[RET:.*]] : tensor<1x1x1x1000xsi32>

  }
}

// -----

module @SingleConvertFPToUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xui32> {
    %res = IE.Convert(%arg0) {dstElemType = ui32}: tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xui32>
    return %res : tensor<1x1x1x1000xui32>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xui32> {
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK: ^bb0([[ARG0:%.+]]: f16, {{%.+}}: i32):
// CHECK: [[OP:%.+]] = arith.fptoui %{{.+}} : f16 to i32
// CHECK: linalg.yield [[OP]] : i32
// CHECK: [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK: return %[[RET:.*]] : tensor<1x1x1x1000xui32>

  }
}

// -----

module @SingleConvertSIToFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xf16> {
    %res = IE.Convert(%arg0) {dstElemType = f16}: tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xf16>
    return %res : tensor<1x1x1x1000xf16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xf16> {
// CHECK: [[RET:%.+]] = tensor.bitcast %arg0 : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RET]] : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK: ^bb0([[ARG0:%.+]]: i32, {{%.+}}: f16):
// CHECK: [[OP:%.+]] = arith.sitofp %{{.+}} : i32 to f16
// CHECK: linalg.yield [[OP]] : f16
// CHECK: return %[[OP:.*]] : tensor<1x1x1x1000xf16>

  }
}

// -----

module @SingleConvertUIToFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xf16> {
    %res = IE.Convert(%arg0) {dstElemType = f16}: tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xf16>
    return %res : tensor<1x1x1x1000xf16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xf16> {
// CHECK: [[RET:%.+]] = tensor.bitcast %arg0 : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RET]] : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK: ^bb0([[ARG0:%.+]]: i32, {{%.+}}: f16):
// CHECK: [[OP:%.+]] = arith.uitofp %{{.+}} : i32 to f16
// CHECK: linalg.yield [[OP]] : f16
// CHECK: return %[[OP:.*]] : tensor<1x1x1x1000xf16>

  }
}

// -----

module @SingleConvertExtFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf32> {
    %res = IE.Convert(%arg0) {dstElemType = f32}: tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
    return %res : tensor<1x1x1x1000xf32>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf32> {
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf32>) {
// CHECK: ^bb0([[ARG0:%.+]]: f16, {{%.+}}: f32):
// CHECK: [[OP:%.+]] = arith.extf %{{.+}} : f16 to f32
// CHECK: linalg.yield [[OP]] : f32
// CHECK: return %[[LINALG_OP:.*]] : tensor<1x1x1x1000xf32>

  }
}

// -----

module @SingleConvertTruncFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf16> {
    %res = IE.Convert(%arg0) {dstElemType = f16}: tensor<1x1x1x1000xf32> -> tensor<1x1x1x1000xf16>
    return %res : tensor<1x1x1x1000xf16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf16> {
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1x1000xf32>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK: ^bb0([[ARG0:%.+]]: f32, {{%.+}}: f16):
// CHECK: [[OP:%.+]] = arith.truncf %{{.+}} : f32 to f16
// CHECK: linalg.yield [[OP]] : f16
// CHECK: return %[[LINALG_OP:.*]] : tensor<1x1x1x1000xf16>

  }
}

// -----

module @SingleConvertExtSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xsi16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi16>) -> tensor<1x1x1x1000xsi32> {
    %res = IE.Convert(%arg0) {dstElemType = si32}: tensor<1x1x1x1000xsi16> -> tensor<1x1x1x1000xsi32>
    return %res : tensor<1x1x1x1000xsi32>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xsi16>) -> tensor<1x1x1x1000xsi32> {
// CHECK: [[RET:%.+]] = tensor.bitcast %arg0 : tensor<1x1x1x1000xsi16> to tensor<1x1x1x1000xi16>
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RET]] : tensor<1x1x1x1000xi16>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK: ^bb0([[ARG0:%.+]]: i16, {{%.+}}: i32):
// CHECK: [[OP:%.+]] = arith.extsi %{{.+}} : i16 to i32
// CHECK: linalg.yield [[OP]] : i32
// CHECK: [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK: return %[[OP:.*]] : tensor<1x1x1x1000xsi32>

  }
}

// -----

module @SingleConvertExtUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xui16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui16>) -> tensor<1x1x1x1000xui32> {
    %res = IE.Convert(%arg0) {dstElemType = ui32}: tensor<1x1x1x1000xui16> -> tensor<1x1x1x1000xui32>
    return %res : tensor<1x1x1x1000xui32>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xui16>) -> tensor<1x1x1x1000xui32> {
// CHECK: [[RET:%.+]] = tensor.bitcast %arg0 : tensor<1x1x1x1000xui16> to tensor<1x1x1x1000xi16>
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RET]] : tensor<1x1x1x1000xi16>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK: ^bb0([[ARG0:%.+]]: i16, {{%.+}}: i32):
// CHECK: [[OP:%.+]] = arith.extui %{{.+}} : i16 to i32
// CHECK: linalg.yield [[OP]] : i32
// CHECK: [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK: return %[[OP:.*]] : tensor<1x1x1x1000xui32>

  }
}

// -----

module @SingleConvertTruncILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xi32>) -> tensor<1x1x1x1000xi16> {
    %res = IE.Convert(%arg0) {dstElemType = i16}: tensor<1x1x1x1000xi32> -> tensor<1x1x1x1000xi16>
    return %res : tensor<1x1x1x1000xi16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xi32>) -> tensor<1x1x1x1000xi16> {
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi16> 
// CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi16>) {
// CHECK: ^bb0([[ARG0:%.+]]: i32, {{%.+}}: i16):
// CHECK: [[OP:%.+]] = arith.trunci %{{.+}} : i32 to i16
// CHECK: linalg.yield [[OP]] : i16
// CHECK: return %[[LINALG_OP:.*]] : tensor<1x1x1x1000xi16>

  }
} 

// -----

module @SingleNoConvertLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xi16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xi16>) -> tensor<1x1x1x1000xi16> {
    %res = IE.Convert(%arg0) {dstElemType = i16}: tensor<1x1x1x1000xi16> -> tensor<1x1x1x1000xi16>
    return %res : tensor<1x1x1x1000xi16>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xi16>) -> tensor<1x1x1x1000xi16> {
// CHECK: return %{{.+}} : tensor<1x1x1x1000xi16>
  }
} 
