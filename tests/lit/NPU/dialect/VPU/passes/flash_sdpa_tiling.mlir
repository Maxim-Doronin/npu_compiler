//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --flash-sdpa-tiling="enable-pipelining=false" %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --flash-sdpa-tiling="enable-pipelining=true" %s | FileCheck %s --check-prefix=PIPELINING
// REQUIRES: arch-NPU5010

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA8kSeqLen
// PIPELINING-LABEL: @FlashSDPA8kSeqLen
func.func @FlashSDPA8kSeqLen(%arg0: tensor<1x8x8192x32xf16>, %arg1: tensor<1x8x128x32xf16>, %arg2: tensor<1x8x128x64xf16>, %arg3: tensor<1x8x8192x128xf16>)
                                  -> tensor<1x8x8192x64xf16> {
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x2x8192x128xf16> = dense<0.000000e+00> : tensor<1x2x8192x128xf16>
    %cst_3 = const.Declare tensor<1x8x8192x1xf32> = dense<0.000000e+00> : tensor<1x8x8192x1xf32>
    %cst_4 = const.Declare tensor<1x8x8192x1xf16> = dense<0xFC00> : tensor<1x8x8192x1xf16>
    %cst_5 = const.Declare tensor<1x8x8192x64xf16> = dense<0.000000e+00> : tensor<1x8x8192x64xf16>

    %value_reordered = IE.Reorder(%arg2) {dstOrder = #NCWH} : tensor<1x8x128x64xf16> -> tensor<1x8x128x64xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x8x8192x32xf16>, tensor<1x8x128x32xf16>, tensor<1x8x128x64xf16, {order = #NCWH}>,
            tensor<1x2x8192x128xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x128x4xsi32>,
            tensor<1x1x64x4xsi32>, tensor<1x8x8192x64xf16>, tensor<1x8x8192x1xf16>,
            tensor<1x8x8192x1xf32>, tensor<1x8x8192x128xf16>
        -> tensor<1x8x8192x64xf16>, tensor<1x8x8192x1xf16>, tensor<1x8x8192x1xf32>

    return %result_running_output : tensor<1x8x8192x64xf16>

    // Disabled pipelining: 2 query tiles per head (4128 + 4064), 16 total ops
    // CHECK-DAG:   [[VALUE_REORDERED:%.+]] = IE.Reorder({{%.+}}) {dstOrder = #NCWH}
    // CHECK:       [[Q0:%.+]] = VPU.Slice [[QUERY:%.+]] [0, 0, 0, 0] [1, 1, 4128, 32] : tensor<1x8x8192x32xf16> to tensor<1x1x4128x32xf16>
    // CHECK:       [[KEY0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 1, 128, 32]
    // CHECK:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 128, 64]
    // CHECK:       [[AUX0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 4128, 128]
    // CHECK:       VPU.FlashSDPA([[Q0]], [[KEY0]], [[VAL0]], [[AUX0]]
    // CHECK-SAME:      kv_num_blocks = 1
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x1x4128x64xf16>, tensor<1x1x4128x1xf16>, tensor<1x1x4128x1xf32>

    // Heads 1 through 7, first query tile
    // CHECK-COUNT-7: -> tensor<1x1x4128x64xf16>

    // Second query tile for head 0
    // CHECK:       [[Q1:%.+]] = VPU.Slice [[QUERY]] [0, 0, 4128, 0] [1, 1, 4064, 32]
    // CHECK:       VPU.FlashSDPA([[Q1]], [[KEY0]], [[VAL0]],
    // CHECK-SAME:      kv_num_blocks = 1
    // CHECK-SAME:      -> tensor<1x1x4064x64xf16>

    // Heads 1 through 7, second query tile
    // CHECK-COUNT-7: -> tensor<1x1x4064x64xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat(
    // CHECK-SAME:      -> tensor<1x8x8192x64xf16>
    // CHECK:       return [[CONCAT]]

    // Enabled pipelining: 3 query tiles per head (2736 + 2736 + 2720), 24 total ops
    // PIPELINING-DAG:   [[VALUE_REORDERED:%.+]] = IE.Reorder({{%.+}}) {dstOrder = #NCWH}
    // PIPELINING:       [[Q0:%.+]] = VPU.Slice [[QUERY:%.+]] [0, 0, 0, 0] [1, 1, 2736, 32] : tensor<1x8x8192x32xf16> to tensor<1x1x2736x32xf16>
    // PIPELINING:       [[KEY0:%.+]] = VPU.Slice [[KEY:%.+]] [0, 0, 0, 0] [1, 1, 128, 32]
    // PIPELINING:       [[VAL0:%.+]] = VPU.Slice [[VALUE_REORDERED]] [0, 0, 0, 0] [1, 1, 128, 64]
    // PIPELINING:       [[AUX0:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 0] [1, 1, 2736, 128]
    // PIPELINING:       VPU.FlashSDPA([[Q0]], [[KEY0]], [[VAL0]], [[AUX0]]
    // PIPELINING-SAME:      kv_num_blocks = 1
    // PIPELINING-SAME:      tiling_loop_index = 0
    // PIPELINING-SAME:      -> tensor<1x1x2736x64xf16>, tensor<1x1x2736x1xf16>, tensor<1x1x2736x1xf32>

    // Heads 1 through 7, first query tile
    // PIPELINING-COUNT-7: -> tensor<1x1x2736x64xf16>

    // Second query tile for head 0
    // PIPELINING:       [[Q1:%.+]] = VPU.Slice [[QUERY]] [0, 0, 2736, 0] [1, 1, 2736, 32]
    // PIPELINING:       VPU.FlashSDPA([[Q1]], [[KEY0]], [[VAL0]],
    // PIPELINING-SAME:      kv_num_blocks = 1
    // PIPELINING-SAME:      -> tensor<1x1x2736x64xf16>

    // Heads 1 through 7, second query tile
    // PIPELINING-COUNT-7: -> tensor<1x1x2736x64xf16>

    // Third query tile for head 0
    // PIPELINING:       [[Q2:%.+]] = VPU.Slice [[QUERY]] [0, 0, 5472, 0] [1, 1, 2720, 32]
    // PIPELINING:       VPU.FlashSDPA([[Q2]], [[KEY0]], [[VAL0]],
    // PIPELINING-SAME:      kv_num_blocks = 1
    // PIPELINING-SAME:      -> tensor<1x1x2720x64xf16>

    // Heads 1 through 7, third query tile
    // PIPELINING-COUNT-7: -> tensor<1x1x2720x64xf16>

    // PIPELINING:       [[CONCAT:%.+]] = VPU.Concat(
    // PIPELINING-SAME:      -> tensor<1x8x8192x64xf16>
    // PIPELINING:       return [[CONCAT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPAKeyValueTilingWithNoQueryTiling
// PIPELINING-LABEL: @FlashSDPAKeyValueTilingWithNoQueryTiling
func.func @FlashSDPAKeyValueTilingWithNoQueryTiling(%arg0: tensor<1x1x1x128xf16>, %arg1: tensor<1x1x4096x128xf16>, %arg2: tensor<1x1x4096x128xf16>, %arg3: tensor<1x1x1x4096xf16>) -> tensor<1x1x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x4096x4xsi32> = dense<0> : tensor<1x1x4096x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x1x1x4096xf16> = dense<0.000000e+00> : tensor<1x1x1x4096xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<0xFC00> : tensor<1x1x1x1xf16>
    %cst_5 = const.Declare tensor<1x1x1x128xf16> = dense<0.000000e+00> : tensor<1x1x1x128xf16>

    %value_reordered = IE.Reorder(%arg2) {dstOrder = #NCWH} : tensor<1x1x4096x128xf16> -> tensor<1x1x4096x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x1x1x128xf16>, tensor<1x1x4096x128xf16>, tensor<1x1x4096x128xf16, {order = #NCWH}>,
            tensor<1x1x1x4096xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x4096x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf32>, tensor<1x1x1x4096xf16>
        -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>

    return %result_running_output : tensor<1x1x1x128xf16>

    // Single tile: no head or query seq tiling needed, only KV blocks > 1
    // Same result with or without pipelining for this small query
    // CHECK:       VPU.FlashSDPA
    // CHECK-SAME:      kv_num_blocks = 2
    // CHECK-SAME:      tiling_loop_index = 0
    // CHECK-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>
    // CHECK:       return

    // PIPELINING:       VPU.FlashSDPA
    // PIPELINING-SAME:      kv_num_blocks = 2
    // PIPELINING-SAME:      tiling_loop_index = 0
    // PIPELINING-SAME:      -> tensor<1x1x1x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf32>
    // PIPELINING:       return
}
