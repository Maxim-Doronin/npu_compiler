//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tile-incremental-sdpa  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @TileIncrementalSdpaWithAttentionMaskNoScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x512x512xf16>
func.func @TileIncrementalSdpaWithAttentionMaskNoScale(%arg0: tensor<8x512x64xf16>, %arg1: tensor<8x512x64xf16>, %arg2: tensor<8x512x64xf16>, %arg3: tensor<8x512x512xf16>) -> tensor<8x512x64xf16> {
    %cst = const.Declare tensor<8x512xf16> = dense<0xFC00> : tensor<8x512xf16>
    %cst_0 = const.Declare tensor<8x512xf16> = dense<0.000000e+00> : tensor<8x512xf16>
    %cst_1 = const.Declare tensor<8x512x64xf16> = dense<0.000000e+00> : tensor<8x512x64xf16>
    %output_running_max, %output_running_sum, %output_partial_output = IE.IncrementalSDPA(%arg0, %arg1, %arg2, %cst, %cst_0, %cst_1, %arg3) {kv_num_blocks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0>} : tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512xf16>, tensor<8x512xf16>, tensor<8x512x64xf16>, tensor<8x512x512xf16> -> tensor<8x512xf16>, tensor<8x512xf16>, tensor<8x512x64xf16>
    %0 = IE.Unsqueeze(%output_running_sum) {axes_value = [2]} : tensor<8x512xf16> -> tensor<8x512x1xf16>
    %1 = IE.Divide(%output_partial_output, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x512x64xf16>, tensor<8x512x1xf16> -> tensor<8x512x64xf16>
    return %1 : tensor<8x512x64xf16>

    // CHECK-DAG:   [[MAX_INITIAL:%.+]] = const.Declare tensor<8x512xf16> = dense<0xFC00> : tensor<8x512xf16>
    // CHECK-DAG:   [[SUM_INITIAL:%.+]] = const.Declare tensor<8x512xf16> = dense<0.000000e+00> : tensor<8x512xf16>
    // CHECK-DAG:   [[OUT_INITIAL:%.+]] = const.Declare tensor<8x512x64xf16> = dense<0.000000e+00> : tensor<8x512x64xf16>

    // CHECK-DAG:   [[KEY_SLICE_0:%.+]] = IE.Slice [[KEY]] [0, 0, 0] [8, 256, 64] : tensor<8x512x64xf16> to tensor<8x256x64xf16>
    // CHECK-DAG:   [[VALUE_SLICE_0:%.+]] = IE.Slice [[VALUE]] [0, 0, 0] [8, 256, 64] : tensor<8x512x64xf16> to tensor<8x256x64xf16>
    // CHECK-DAG:   [[ATTENTION_MASK_SLICE_0:%.+]] = IE.Slice [[ATTENTION_MASK]] [0, 0, 0] [8, 512, 256] : tensor<8x512x512xf16> to tensor<8x512x256xf16>

    // CHECK:       [[RESULT_MAX_0:%[^, ]+]], [[RESULT_SUM_0:%[^, ]+]], [[RESULT_OUT_0:%[^, ]+]] =
    // CHECK-SAME:      IE.IncrementalSDPA([[QUERY]], [[KEY_SLICE_0]], [[VALUE_SLICE_0]], [[MAX_INITIAL]],
    // CHECK-SAME:                         [[SUM_INITIAL]], [[OUT_INITIAL]], [[ATTENTION_MASK_SLICE_0]])
    // CHECK-SAME:          -> tensor<8x512xf16>, tensor<8x512xf16>, tensor<8x512x64xf16>

    // CHECK-DAG:   [[KEY_SLICE_1:%.+]] = IE.Slice [[KEY]] [0, 256, 0] [8, 256, 64] : tensor<8x512x64xf16> to tensor<8x256x64xf16>
    // CHECK-DAG:   [[VALUE_SLICE_1:%.+]] = IE.Slice [[VALUE]] [0, 256, 0] [8, 256, 64] : tensor<8x512x64xf16> to tensor<8x256x64xf16>
    // CHECK-DAG:   [[ATTENTION_MASK_SLICE_1:%.+]] = IE.Slice [[ATTENTION_MASK]] [0, 0, 256] [8, 512, 256] : tensor<8x512x512xf16> to tensor<8x512x256xf16>

    // CHECK:       [[RESULT_MAX_1:%[^, ]+]], [[RESULT_SUM_1:%[^, ]+]], [[RESULT_OUT_1:%[^, ]+]] =
    // CHECK-SAME:      IE.IncrementalSDPA([[QUERY]], [[KEY_SLICE_1]], [[VALUE_SLICE_1]], [[RESULT_MAX_0]],
    // CHECK-SAME:                         [[RESULT_SUM_0]], [[RESULT_OUT_0]], [[ATTENTION_MASK_SLICE_1]])
    // CHECK-SAME:          -> tensor<8x512xf16>, tensor<8x512xf16>, tensor<8x512x64xf16>

    // CHECK:       [[UNSQUEEZED_SUM:%.+]] = IE.Unsqueeze([[RESULT_SUM_1]])
    // CHECK-SAME:      -> tensor<8x512x1xf16>

    // CHECK:       [[RESULT:%.+]] = IE.Divide([[RESULT_OUT_1]], [[UNSQUEEZED_SUM]])
    // CHECK-SAME:      -> tensor<8x512x64xf16>

    // CHECK:       return [[RESULT]]
}

