//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SWEltwiseNeedAlignmentSOW
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x3x1x255xf16>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<1x3x1x1xf16>
func.func @SWEltwiseNeedAlignmentSOW(%arg0: tensor<1x3x1x255xf16>, %arg1: tensor<1x3x1x1xf16>) -> tensor<1x3x1x255xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} : tensor<1x3x1x255xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x1x255xf16>
    return %0 : tensor<1x3x1x255xf16>

    // CHECK:     [[DATA0:%.+]] = VPU.UnrolledType([[INPUT0]] : tensor<1x3x1x255xf16>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x3x1x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, alignment = [1, 1, 1, 32], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]}>

    // CHECK:     [[DATA1:%.+]] = VPU.UnrolledType([[INPUT1]] : tensor<1x3x1x1xf16>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 1, 1], [1, 3, 1, 1], [1, 3, 1, 1]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 1, 1], [1, 3, 1, 1], [1, 3, 1, 1]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:     [[MUL:%.+]] = VPU.Multiply([[DATA0]], [[DATA1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x3x1x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, alignment = [1, 1, 1, 32], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]}>

    // CHECK:     [[OUT:%.+]] = VPU.UnrolledType([[MUL]] : !VPU.DistributedTensor<1x3x1x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, alignment = [1, 1, 1, 32], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]}>)
    // CHECK-SAME:           -> tensor<1x3x1x255xf16>

    // CHECK:     return [[OUT]] : tensor<1x3x1x255xf16>
}

}
