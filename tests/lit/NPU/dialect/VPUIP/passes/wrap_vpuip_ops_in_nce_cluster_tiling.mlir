//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wrap-vpuip-ops-in-cluster-tiling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInBuf = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @DistributedCopy2CopyOp
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
func.func @DistributedCopy2CopyOp(%arg0: memref<1x32x16x16xf16, #NHWC, @DDR>) -> memref<1x32x16x16xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !DistributedInBuf

    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x16x16xf16, #NHWC, @DDR>)
        outputs(%0 : !DistributedInBuf)
        -> !DistributedInBuf
    %alloc = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @DDR>

    %2 = VPUIP.Copy
        inputs(%1 : !DistributedInBuf)
        outputs(%alloc : memref<1x32x16x16xf16, #NHWC, @DDR>)
        -> memref<1x32x16x16xf16, #NHWC, @DDR>

    return %2 : memref<1x32x16x16xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[COPY_RES0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[ARG0]] as [[INNER0:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[ALLOC0]] as [[INNER_OUT0:%.+]]:  memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer
    // CHECK:                 [[INNER_C0:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[INNER0]]

    // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_RES1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[COPY_RES0]] as [[INNER1:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[ALLOC1]] as [[INNER_OUT1:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK:                 [[INNER_C1:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[INNER1]]

    // CHECK:   return [[COPY_RES1]] : memref<1x32x16x16xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.78431372549019607>

!DistributedBuf = !VPUIP.DistributedBuffer<
    1x32x16x320xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    memory_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]
}>

!DistributedQuantBuf = !VPUIP.DistributedBuffer<
    1x32x16x320x!qElemType, #NWCH, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    memory_shapes = [[1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320], [1, 32, 4, 320]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]
}>

// CHECK-LABEL: @SuperdenseNCEEltwise
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x32x16x320xf16, #NHWC>)
func.func @SuperdenseNCEEltwise(%arg0: memref<1x32x16x320xf16, #NHWC>) -> memref<1x32x16x320x!qElemType, #NWCH> {
    %0 = VPURT.AllocDistributed -> !DistributedBuf

    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x16x320xf16, #NHWC>)
        outputs(%0 : !DistributedBuf)
        -> !DistributedBuf

    %2 = VPURT.AllocDistributed -> !DistributedQuantBuf

    %3 = VPUIP.NCEClusterTask
            {eltwise_type = #VPU.eltwise_type<ADD>, is_superdense, minimumHardwareExecutionCost = 3052 : i64,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%1 : !DistributedBuf)
        weights(%1 : !DistributedBuf)
        parent_input(%1 : !DistributedBuf)
        parent_output(%2 : !DistributedQuantBuf)
        outputs(%2 : !DistributedQuantBuf)
        -> !DistributedQuantBuf
        variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [319, 3, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [319, 3, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [319, 3, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [319, 3, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [319, 3, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [319, 3, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [319, 3, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [319, 3, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %alloc = memref.alloc() : memref<1x32x16x320x!qElemType, #NWCH>

    %4 = VPUIP.Copy inputs(%3 : !DistributedQuantBuf)
        outputs(%alloc : memref<1x32x16x320x!qElemType, #NWCH>) -> memref<1x32x16x320x!qElemType, #NWCH>

    return %4 : memref<1x32x16x320x!qElemType, #NWCH>

    // CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[CT_COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[ARG0]] as [[INNER0:[^:]+]]: memref<1x32x16x320xf16, #NHWC>)
    // CHECK-SAME:       outputs([[ALLOC0]] as [[INNER_OUT0:[^:]+]]: memref<1x32x16x320xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer
    // CHECK:                 [[INNER_C0:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[INNER0]]
    // CHECK-SAME:              outputs([[INNER_OUT0]]

    // CHECK:       [[ALLOC1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[CT_CLUSTER_TASK:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[CT_COPY0]] as [[INNER1:[^:]+]]: memref<1x32x16x320xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:       [[CT_COPY0]] as [[INNER2:[^:]+]]: memref<1x32x16x320xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:       [[CT_COPY0]] as [[INNER3:[^:]+]]: memref<1x32x16x320xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:       [[ALLOC1]] as [[INNER4:[^:]+]]: memref<1x32x16x320x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  outputs([[ALLOC1]] as [[INNER_OUT1:[^:]+]]: memref<1x32x16x320x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer
    // CHECK:                 [[INNER_T1:%.+]] = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_superdense,
    // CHECK-SAME:              input([[INNER1]] : memref<1x32x16x320xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:              weights([[INNER2]] : memref<1x32x16x320xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:              parent_input([[INNER3]] : memref<1x32x16x320xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:              parent_output([[INNER4]] : memref<1x32x16x320x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:              outputs([[INNER_OUT1]] : memref<1x32x16x320x!qElemType, #NWCH, @CMX_NN>)

    // CHECK:       [[ALLOC2:%.+]] = memref.alloc() :  memref<1x32x16x320x!qElemType, #NWCH>

    // CHECK:       [[CT_COPY2:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[CT_CLUSTER_TASK]] as [[INNER5:[^:]+]]
    // CHECK-SAME:       outputs([[ALLOC2]] as [[INNER_OUT2:[^:]+]]
    // CHECK-SAME:       -> memref<1x32x16x320x!qElemType, #NWCH> {
    // CHECK:                 [[INNER_C2:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[INNER5]]
    // CHECK-SAME:              outputs([[INNER_OUT2]]

    // CHECK:       return [[CT_COPY2]] : memref<1x32x16x320x!qElemType, #NWCH>
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

!InputBuffer = !VPUIP.DistributedBuffer<
    1x3x256x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [2, 1],
    num_clusters = 2 : i64
}>

!OutputBuffer = !VPUIP.DistributedBuffer<
    1x4x256x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [2, 1],
    num_clusters = 2,
    equal_memory_and_compute_view
}>

// CHECK-LABEL: @NcePermute_MultiTile
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x3x256x224xf16, #NCHW, @CMX_NN,
// CHECK-SAME: )
// CHECK-SAME: -> !VPUIP.DistributedBuffer<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
func.func @NcePermute_MultiTile(%in: !InputBuffer) -> !OutputBuffer {
    %0 = VPUIP.ViewOp %in : !InputBuffer
        to !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [2, 1], num_clusters = 2 : i64}>

    %1 = VPURT.AllocDistributed 
        -> !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    %2 = VPUIP.NCEClusterTask {is_permute_quantize, is_superdense, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%0 : !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [2, 1], num_clusters = 2 : i64}>)
        weights(%0 : !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [2, 1], num_clusters = 2 : i64}>)
        parent_input(%0 : !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [2, 1], num_clusters = 2 : i64}>)
        parent_output(%1 : !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}>)
        outputs(%1 : !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}>)
        -> !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}> variants : {
      DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [255, 2, 223], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <ADD>, clamp_low = 0 : i64, clamp_high = 255 : i64,
               lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01], fp_prelu_alpha = 1.000000e+00 : f64>}
    }

    %3 = VPUIP.ViewOp %2 : !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
                {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
                pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}>
        to !OutputBuffer

    return %3 : !OutputBuffer

    // CHECK:       [[VIEW_OP_IN:%.*]] = VPUIP.ViewOp [[ARG0]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,

    // CHECK:       [[OUT_BUF:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:   {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
    // CHECK-SAME:   pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:   strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    // CHECK:        [[MULTI_TILE_RES:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[VIEW_OP_IN]] as [[INNER0:[^:]+]]: memref<1x224x3x256xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:       [[VIEW_OP_IN]] as [[INNER1:[^:]+]]: memref<1x224x3x256xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:       [[VIEW_OP_IN]] as [[INNER2:[^:]+]]: memref<1x224x3x256xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:       [[OUT_BUF]] as [[INNER3:[^:]+]]: memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] as [[INNER_OUT0:[^:]+]]: memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN
    // CHECK:           [[INNER_CT:%.+]] = VPUIP.NCEClusterTask {
    // CHECK-SAME:          is_permute_quantize
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:          input([[INNER0]] : memref<1x224x3x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          weights([[INNER1]] : memref<1x224x3x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_input([[INNER2]] : memref<1x224x3x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[INNER3]] : memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:          outputs([[INNER_OUT0]] : memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:     -> memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>

    // CHECK:       [[RES:%.+]] = VPUIP.ViewOp [[MULTI_TILE_RES]] :
    // CHECK-SAME:   !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:   to
    // CHECK-SAME:   !VPUIP.DistributedBuffer<1x4x256x224x!qElemType, #NHWC, @CMX_NN,

    // CHECK: return [[RES]] : !VPUIP.DistributedBuffer<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// module @executors {
// IE.TileResource 6 of @NCE at 1.700000e+03 MHz

!InputBuffer0 = !VPUIP.DistributedBuffer<
    1x32x44x44xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]
}>

!InputBuffer1 = !VPUIP.DistributedBuffer<
    1x1x44x44xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]
}>

!OutputBuffer = !VPUIP.DistributedBuffer<
    1x32x44x44xi8, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]
}>

// CHECK-LABEL: @EqualSWSOHTileNotAtBroadcastAxis
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x32x44x44xf16, #NCHW, @CMX_NN,
// CHECK-SAME: [[ARG1:%.+]]: !VPUIP.DistributedBuffer<1x1x44x44xf16, #NCHW, @CMX_NN,
// CHECK-SAME: -> !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN,
func.func @EqualSWSOHTileNotAtBroadcastAxis(%arg0: !InputBuffer0, %arg1: !InputBuffer1) -> !OutputBuffer {

    %0 = VPURT.AllocDistributed -> !OutputBuffer

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Equal
    inputs(%arg0 as %arg2: !InputBuffer0, %arg1 as %arg3: !InputBuffer1)
    outputs(%0 as %arg4: !OutputBuffer) on tile 0
     -> !OutputBuffer{
      VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4) : !InputBuffer0, !InputBuffer1, !OutputBuffer
    }

    return %results : !OutputBuffer

// CHECK:        [[ALLOC_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED",

// CHECK:       [[CLUSTER_OUT:%.+]] = VPUIP.NCEClusterTiling
// CHECK-SAME:   inputs([[ARG0]] as [[INNER0:[^:]+]]: memref<1x32x44x44xf16, @CMX_NN>, [[ARG1]] as [[INNER1:[^:]+]]: memref<1x1x44x44xf16, @CMX_NN>)
// CHECK-SAME:   outputs([[ALLOC_OUT]] as [[INNER_OUT:[^:]+]]: memref<1x32x44x44xi8, @CMX_NN>)
// CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN,
// CHECK-SAME:                 {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:        compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:        memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
// CHECK:           [[INNER_SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Equal
// CHECK-SAME:      inputs([[INNER0]] as [[SWK0:[^:]+]]: memref<1x32x44x44xf16, @CMX_NN>,
// CHECK-SAME:      [[INNER1]] as [[SWK1:[^:]+]]: memref<1x1x44x44xf16, @CMX_NN>)
// CHECK-SAME:      outputs([[INNER_OUT]] as [[SWK2:[^:]+]]: memref<1x32x44x44xi8, @CMX_NN>)
// CHECK-SAME:      on tile 0 -> memref<1x32x44x44xi8, @CMX_NN>{
// CHECK:                 VPUIP.SW.Kernel.run([[SWK0]], [[SWK1]], [[SWK2]]) : memref<1x32x44x44xf16, @CMX_NN>, memref<1x1x44x44xf16, @CMX_NN>, memref<1x32x44x44xi8, @CMX_NN>
// CHECK:         }
// CHECK:      }
// CHECK:    return [[CLUSTER_OUT]] : !VPUIP.DistributedBuffer<1x32x44x44xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64

}
