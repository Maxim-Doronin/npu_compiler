//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --reduce-num-tiles-for-small-models %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @NoMultiplyNumClustersRemained
module @NoMultiplyNumClustersRemained {
    IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
        IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
        IE.ExecutorResource 2 of @SHAVE_ACT
        IE.ExecutorResource 1 of @DPU
    }
    IE.ExecutorResource 2 of @DMA_NN
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1x55x55xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x55x128xf16>
    }

    func.func @main(%input: tensor<1x1x55x55xf16>) -> tensor<1x1x55x128xf16> {
        %weights = const.Declare tensor<1x1x128x55xf16> = dense<1.0> : tensor<1x1x128x55xf16>
        %matmul = IE.MatMul(%input, %weights) {transpose_b} : tensor<1x1x55x55xf16>, tensor<1x1x128x55xf16> -> tensor<1x1x55x128xf16>
        %softmax = IE.SoftMax(%matmul) {axisInd = 3 : i64} : tensor<1x1x55x128xf16> -> tensor<1x1x55x128xf16>
        return %softmax : tensor<1x1x55x128xf16>
    }
}

// CHECK: IE.TileResource 6 of @NCE
// CHECK: IE.ExecutorResource 2 of @DMA_NN

// -----

// CHECK-LABEL: @NoMatMulNumClustersRemained
module @NoMatMulNumClustersRemained {
    IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
        IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
        IE.ExecutorResource 2 of @SHAVE_ACT
        IE.ExecutorResource 1 of @DPU
    }
    IE.ExecutorResource 2 of @DMA_NN
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1x55x128xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x55x128xf16>
    }

    func.func @main(%input: tensor<1x1x55x128xf16>) -> tensor<1x1x55x128xf16> {
        %scale = const.Declare tensor<1x1x55x128xf16> = dense<1.0> : tensor<1x1x55x128xf16>
        %multiply = IE.Multiply(%input, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x55x128xf16>, tensor<1x1x55x128xf16> -> tensor<1x1x55x128xf16>
        %softmax = IE.SoftMax(%multiply) {axisInd = 3 : i64} : tensor<1x1x55x128xf16> -> tensor<1x1x55x128xf16>
        return %softmax : tensor<1x1x55x128xf16>
    }
}

// CHECK: IE.TileResource 6 of @NCE
// CHECK: IE.ExecutorResource 2 of @DMA_NN

// -----

// CHECK-LABEL: @NoSoftMaxNumClustersRemained
module @NoSoftMaxNumClustersRemained {
    IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
        IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
        IE.ExecutorResource 2 of @SHAVE_ACT
        IE.ExecutorResource 1 of @DPU
    }
    IE.ExecutorResource 2 of @DMA_NN
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1x55x55xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x55x128xf16>
    }

    func.func @main(%input: tensor<1x1x55x55xf16>) -> tensor<1x1x55x128xf16> {
        %scale = const.Declare tensor<1x1x55x128xf16> = dense<1.0> : tensor<1x1x55x128xf16>
        %weights = const.Declare tensor<1x1x128x55xf16> = dense<1.0> : tensor<1x1x128x55xf16>
        %matmul = IE.MatMul(%input, %weights) {transpose_b} : tensor<1x1x55x55xf16>, tensor<1x1x128x55xf16> -> tensor<1x1x55x128xf16>
        %multiply = IE.Multiply(%matmul, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x55x128xf16>, tensor<1x1x55x128xf16> -> tensor<1x1x55x128xf16>
        return %multiply : tensor<1x1x55x128xf16>
    }
}

// CHECK: IE.TileResource 6 of @NCE
// CHECK: IE.ExecutorResource 2 of @DMA_NN

// -----

// CHECK-LABEL: @MatMulMultiplySoftMaxNumClustersReduced
module @MatMulMultiplySoftMaxNumClustersReduced {
    IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
        IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
        IE.ExecutorResource 2 of @SHAVE_ACT
        IE.ExecutorResource 1 of @DPU
    }
    IE.ExecutorResource 2 of @DMA_NN
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1x55x55xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x55x128xf16>
    }

    func.func @main(%input: tensor<1x1x55x55xf16>) -> tensor<1x1x55x128xf16> {
        %scale = const.Declare tensor<1x1x55x128xf16> = dense<1.0> : tensor<1x1x55x128xf16>
        %weights = const.Declare tensor<1x1x128x55xf16> = dense<1.0> : tensor<1x1x128x55xf16>
        %matmul = IE.MatMul(%input, %weights) {transpose_b} : tensor<1x1x55x55xf16>, tensor<1x1x128x55xf16> -> tensor<1x1x55x128xf16>
        %multiply = IE.Multiply(%matmul, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x55x128xf16>, tensor<1x1x55x128xf16> -> tensor<1x1x55x128xf16>
        %softmax = IE.SoftMax(%multiply) {axisInd = 3 : i64} : tensor<1x1x55x128xf16> -> tensor<1x1x55x128xf16>
        return %softmax : tensor<1x1x55x128xf16>
    }
}

// CHECK: IE.TileResource 1 of @NCE
// CHECK: IE.ExecutorResource 1 of @DMA_NN

// -----

// CHECK-LABEL: @BigShapesNumClustersRemained
module @BigShapesNumClustersRemained {
    IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
        IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
        IE.ExecutorResource 2 of @SHAVE_ACT
        IE.ExecutorResource 1 of @DPU
    }
    IE.ExecutorResource 2 of @DMA_NN
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1x555x555xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x555x128xf16>
    }

    func.func @main(%input: tensor<1x1x555x555xf16>) -> tensor<1x1x555x128xf16> {
        %scale = const.Declare tensor<1x1x555x128xf16> = dense<1.0> : tensor<1x1x555x128xf16>
        %weights = const.Declare tensor<1x1x128x555xf16> = dense<1.0> : tensor<1x1x128x555xf16>
        %matmul = IE.MatMul(%input, %weights) {transpose_b} : tensor<1x1x555x555xf16>, tensor<1x1x128x555xf16> -> tensor<1x1x555x128xf16>
        %multiply = IE.Multiply(%matmul, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x555x128xf16>, tensor<1x1x555x128xf16> -> tensor<1x1x555x128xf16>
        %softmax = IE.SoftMax(%multiply) {axisInd = 3 : i64} : tensor<1x1x555x128xf16> -> tensor<1x1x555x128xf16>
        return %softmax : tensor<1x1x555x128xf16>
    }
}

// CHECK: IE.TileResource 6 of @NCE
// CHECK: IE.ExecutorResource 2 of @DMA_NN
