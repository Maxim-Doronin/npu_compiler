//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --adjust-distributed-tensor-around-ops %s | FileCheck %s
// REQUIRES: platform-NPU5010

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// H-tiled OVERLAPPED producer with tight memory (no halos).
// Slice takes H[3..17), userCopy output has 1-row halos.
// Expected: producer output memory shapes expand to cover the slice destination.
//   memory_shapes: [[1,64,10,16],[1,64,10,16]] -> [[1,64,11,16],[1,64,11,16]]
//   memory_offsets: [[0,0,0,0],[0,0,10,0]]    -> [[0,0,0,0],[0,0,9,0]]

!HProducerInput = !VPU.DistributedTensor<
    1x32x20x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]],
    memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!HProducerOutputTight = !VPU.DistributedTensor<
    1x64x20x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]],
    memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!HProducerFilter = !VPU.DistributedTensor<
    64x32x1x1xi8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 32, 1, 1], [64, 32, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 32, 1, 1], [64, 32, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// Slice H[3..17) -> 14 rows, with 1-row halos for each cluster.
!HSliceDest = !VPU.DistributedTensor<
    1x64x14x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 7, 16], [1, 64, 7, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 64, 8, 16], [1, 64, 8, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!HConsumerFilter = !VPU.DistributedTensor<
    32x64x1x1xi8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[32, 64, 1, 1], [32, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 64, 1, 1], [32, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!HConsumerOutput = !VPU.DistributedTensor<
    1x32x14x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 7, 16], [1, 32, 7, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 32, 7, 16], [1, 32, 7, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]
}>

// CHECK-LABEL: @HaloAssistedSliceOptimizationHDim
func.func @HaloAssistedSliceOptimizationHDim(
        %input: !HProducerInput,
        %filter0: !HProducerFilter,
        %filter1: !HConsumerFilter) -> !HConsumerOutput {
    %conv0 = VPU.NCE.Convolution(%input, %filter0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [64, 32, 1, 1], strides = [1, 1],
        vf_loop_layer_index = 0 : i64
    } : !HProducerInput, !HProducerFilter -> !HProducerOutputTight

    %copy0 = VPU.Copy(%conv0) : !HProducerOutputTight -> tensor<1x64x20x16xi8, {order = #NHWC}>

    %slice = VPU.Slice %copy0 [0, 0, 3, 0] [1, 64, 14, 16]
        : tensor<1x64x20x16xi8, {order = #NHWC}> to tensor<1x64x14x16xi8, {order = #NHWC}>

    %copy1 = VPU.Copy(%slice) {out_mem_space = @CMX_NN}
        : tensor<1x64x14x16xi8, {order = #NHWC}> -> !HSliceDest

    %conv1 = VPU.NCE.Convolution(%copy1, %filter1) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [32, 64, 1, 1], strides = [1, 1],
        vf_loop_layer_index = 0 : i64
    } : !HSliceDest, !HConsumerFilter -> !HConsumerOutput

    return %conv1 : !HConsumerOutput

    // Producer output memory expanded on H to cover the halo rows required by the slice destination.
    // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 11, 16], [1, 64, 11, 16]]
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0]]
    // CHECK-NOT{LITERAL}: memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// W-tiled OVERLAPPED producer with tight memory (no halos).
// Slice takes W[2..14), userCopy output has 1-col halos.
// Expected: producer output memory shapes expand to cover the slice destination on W.
//   memory_shapes: [[1,64,20,8],[1,64,20,8]] -> [[1,64,20,9],[1,64,20,9]]
//   memory_offsets: [[0,0,0,0],[0,0,0,8]]    -> [[0,0,0,0],[0,0,0,7]]

!WProducerInput = !VPU.DistributedTensor<
    1x64x20x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 20, 8], [1, 64, 20, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    memory_shapes = [[1, 64, 20, 8], [1, 64, 20, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]]
}>

!WProducerOutputTight = !VPU.DistributedTensor<
    1x64x20x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 20, 8], [1, 64, 20, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    memory_shapes = [[1, 64, 20, 8], [1, 64, 20, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]]
}>

!WProducerFilter = !VPU.DistributedTensor<
    64x64x1x1xi8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 64, 1, 1], [64, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 64, 1, 1], [64, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// Slice W[2..14) -> 12 cols, with 1-col halos for each cluster.
!WSliceDest = !VPU.DistributedTensor<
    1x64x20x12xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 20, 6], [1, 64, 20, 6]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]],
    memory_shapes = [[1, 64, 20, 7], [1, 64, 20, 7]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 5]]
}>

!WConsumerFilter = !VPU.DistributedTensor<
    32x64x1x1xi8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[32, 64, 1, 1], [32, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 64, 1, 1], [32, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WConsumerOutput = !VPU.DistributedTensor<
    1x32x20x12xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 20, 6], [1, 32, 20, 6]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]],
    memory_shapes = [[1, 32, 20, 6], [1, 32, 20, 6]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]
}>

// CHECK-LABEL: @HaloAssistedSliceOptimizationWDim
func.func @HaloAssistedSliceOptimizationWDim(
        %input: !WProducerInput,
        %filter0: !WProducerFilter,
        %filter1: !WConsumerFilter) -> !WConsumerOutput {
    %conv0 = VPU.NCE.Convolution(%input, %filter0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [64, 64, 1, 1], strides = [1, 1],
        vf_loop_layer_index = 0 : i64
    } : !WProducerInput, !WProducerFilter -> !WProducerOutputTight

    %copy0 = VPU.Copy(%conv0) : !WProducerOutputTight -> tensor<1x64x20x16xi8, {order = #NHWC}>

    %slice = VPU.Slice %copy0 [0, 0, 0, 2] [1, 64, 20, 12]
        : tensor<1x64x20x16xi8, {order = #NHWC}> to tensor<1x64x20x12xi8, {order = #NHWC}>

    %copy1 = VPU.Copy(%slice) {out_mem_space = @CMX_NN}
        : tensor<1x64x20x12xi8, {order = #NHWC}> -> !WSliceDest

    %conv1 = VPU.NCE.Convolution(%copy1, %filter1) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [32, 64, 1, 1], strides = [1, 1],
        vf_loop_layer_index = 0 : i64
    } : !WSliceDest, !WConsumerFilter -> !WConsumerOutput

    return %conv1 : !WConsumerOutput

    // Producer output memory expanded on W to cover the halo cols required by the slice destination.
    // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 20, 9], [1, 64, 20, 9]]
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 7]]
    // CHECK-NOT{LITERAL}: memory_shapes = [[1, 64, 20, 8], [1, 64, 20, 8]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// No vf_loop_layer_index on the producer NCE op.
// HaloAssistedSliceOptimization requires this attribute on both producer and consumer ops.
// Pass must skip; producer output type must remain unchanged (tight memory shapes).

!NegProducerInput = !VPU.DistributedTensor<
    1x32x20x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]],
    memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!NegProducerOutputTight = !VPU.DistributedTensor<
    1x64x20x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]],
    memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!NegProducerFilter = !VPU.DistributedTensor<
    64x32x1x1xi8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 32, 1, 1], [64, 32, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 32, 1, 1], [64, 32, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!NegSliceDest = !VPU.DistributedTensor<
    1x64x14x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 7, 16], [1, 64, 7, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 64, 8, 16], [1, 64, 8, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!NegConsumerFilter = !VPU.DistributedTensor<
    32x64x1x1xi8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[32, 64, 1, 1], [32, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 64, 1, 1], [32, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!NegConsumerOutput = !VPU.DistributedTensor<
    1x32x14x16xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 7, 16], [1, 32, 7, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 32, 7, 16], [1, 32, 7, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]
}>

// CHECK-LABEL: @HaloAssistedSliceSkippedNoVFAttr
func.func @HaloAssistedSliceSkippedNoVFAttr(
        %input: !NegProducerInput,
        %filter0: !NegProducerFilter,
        %filter1: !NegConsumerFilter) -> !NegConsumerOutput {
    // Producer has no vf_loop_layer_index — optimization must not fire.
    %conv0 = VPU.NCE.Convolution(%input, %filter0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [64, 32, 1, 1], strides = [1, 1]
    } : !NegProducerInput, !NegProducerFilter -> !NegProducerOutputTight

    %copy0 = VPU.Copy(%conv0) : !NegProducerOutputTight -> tensor<1x64x20x16xi8, {order = #NHWC}>

    %slice = VPU.Slice %copy0 [0, 0, 3, 0] [1, 64, 14, 16]
        : tensor<1x64x20x16xi8, {order = #NHWC}> to tensor<1x64x14x16xi8, {order = #NHWC}>

    %copy1 = VPU.Copy(%slice) {out_mem_space = @CMX_NN}
        : tensor<1x64x14x16xi8, {order = #NHWC}> -> !NegSliceDest

    %conv1 = VPU.NCE.Convolution(%copy1, %filter1) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [32, 64, 1, 1], strides = [1, 1],
        vf_loop_layer_index = 0 : i64
    } : !NegSliceDest, !NegConsumerFilter -> !NegConsumerOutput

    return %conv1 : !NegConsumerOutput

    // Producer output type must be unchanged: tight memory shapes are preserved.
    // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]]
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}
