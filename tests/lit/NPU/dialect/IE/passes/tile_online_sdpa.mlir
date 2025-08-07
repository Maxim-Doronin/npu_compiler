//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tile-online-sdpa  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @TileOnlineSdpaWithAttentionMaskNoScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x512x512xf16>
func.func @TileOnlineSdpaWithAttentionMaskNoScale(%arg0: tensor<8x512x64xf16>, %arg1: tensor<8x512x64xf16>, %arg2: tensor<8x512x64xf16>, %arg3: tensor<8x512x512xf16>) -> tensor<8x512x64xf16> {
    %0 = IE.OnlineSDPA(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x512xf16> -> tensor<8x512x64xf16>
    return %0 : tensor<8x512x64xf16>

    // CHECK-DAG:   [[QUERY_SLICE_0:%.+]] = IE.Slice [[QUERY]] [0, 0, 0] [8, 256, 64] : tensor<8x512x64xf16> to tensor<8x256x64xf16>
    // CHECK-DAG:   [[ATTENTION_MASK_SLICE_0:%.+]] = IE.Slice [[ATTENTION_MASK]] [0, 0, 0] [8, 256, 512] : tensor<8x512x512xf16> to tensor<8x256x512xf16>
    // CHECK:       [[SDPA_0:%.+]] = IE.OnlineSDPA([[QUERY_SLICE_0]], [[KEY]], [[VALUE]], [[ATTENTION_MASK_SLICE_0]])
    // CHECK-SAME:      kv_num_blocks = 2
    // CHECK-SAME:      -> tensor<8x256x64xf16>

    // CHECK-DAG:   [[QUERY_SLICE_1:%.+]] = IE.Slice [[QUERY]] [0, 256, 0] [8, 256, 64] : tensor<8x512x64xf16> to tensor<8x256x64xf16>
    // CHECK-DAG:   [[ATTENTION_MASK_SLICE_1:%.+]] = IE.Slice [[ATTENTION_MASK]] [0, 256, 0] [8, 256, 512] : tensor<8x512x512xf16> to tensor<8x256x512xf16>
    // CHECK:       [[SDPA_1:%.+]] = IE.OnlineSDPA([[QUERY_SLICE_1]], [[KEY]], [[VALUE]], [[ATTENTION_MASK_SLICE_1]])
    // CHECK-SAME:      kv_num_blocks = 2
    // CHECK-SAME:      -> tensor<8x256x64xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SDPA_0]], [[SDPA_1]])
    // CHECK-SAME:      -> tensor<8x512x64xf16>

    // CHECK:       return [[CONCAT]]
}

