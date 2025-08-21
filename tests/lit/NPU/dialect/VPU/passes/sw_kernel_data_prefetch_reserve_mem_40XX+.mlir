//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --sw-kernel-data-prefetch-reserve-mem %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @SimpleGraph {
  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

    // reserve dummy memory at the end of CMX

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         SWKernelPrefetchingReservedMemory
    // CHECK:           IE.MemoryResource 1024 bytes of @CMX_NN offset 1473536
}

// -----

module @SimpleGraphWithReservedMem {
  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @CustomReservedMemory {
            IE.MemoryResource 512 bytes of @CMX_NN offset 1474048
        }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

    // Reserve additional memory

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         SWKernelPrefetchingReservedMemory
    // CHECK:           IE.MemoryResource 512 bytes of @CMX_NN offset 1473536
    // CHECK:         CustomReservedMemory
    // CHECK:           IE.MemoryResource 512 bytes of @CMX_NN offset 1474048
}

// -----

module @SimpleGraphWithReservedMemHasEnoughSize {
  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @CustomReservedMemory {
            IE.MemoryResource 1024 bytes of @CMX_NN offset 1473536
        }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

    // no need to change the reserved memory size, just put it at the end of CMX

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         CustomReservedMemory
    // CHECK:           IE.MemoryResource 1024 bytes of @CMX_NN offset 1473536
}

// -----

module @SimpleGraphWith2ReservedMem {
  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @CustomReservedMemory1 {
            IE.MemoryResource 512 bytes of @CMX_NN offset 1473984
        }

        module @CustomReservedMemory2 {
            IE.MemoryResource 64 bytes of @CMX_NN offset 1474496
        }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

    // Reserve missing chunk of memory

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         SWKernelPrefetchingReservedMemory
    // CHECK:           IE.MemoryResource 448 bytes of @CMX_NN offset 1473536
    // CHECK:         CustomReservedMemory1
    // CHECK:           IE.MemoryResource 512 bytes of @CMX_NN offset 1473984
    // CHECK:         CustomReservedMemory2
    // CHECK:           IE.MemoryResource 64 bytes of @CMX_NN offset 1474496
}

// -----

module @SimpleGraphWith2ReservedMemHaveEnoughTotalSize {
  module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    builtin.module @ReservedMemory {
        module @CustomReservedMemory1 {
            IE.MemoryResource 128 bytes of @CMX_NN offset 1473536
        }

        module @CustomReservedMemory2 {
            IE.MemoryResource 896 bytes of @CMX_NN offset 1473664
        }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }

  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    %results = VPU.Gelu(%arg0) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    return %results: tensor<1x16x4x4xf16>
  }

    // CHECK:     IE.TileResource
    // CHECK:       ReservedMemory
    // CHECK:         CustomReservedMemory1
    // CHECK:           IE.MemoryResource 128 bytes of @CMX_NN offset 1473536
    // CHECK:         CustomReservedMemory2
    // CHECK:           IE.MemoryResource 896 bytes of @CMX_NN offset 1473664
}

// -----

module @SimpleGraphNoSWKernel {

  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  }
  func.func @main(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    return %arg0 : tensor<1x16x4x4xf16>
  }
    // not change if no SW Kernel

    // CHECK-NOT:     ReservedMemory
}
