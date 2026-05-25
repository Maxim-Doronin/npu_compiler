//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux;

namespace vpux::VPU::arch40xx {

bool isSmallKernelOptimizationSupported(mlir::Operation* /*op*/, ArrayRef<VPU::DPUWorkloadOp> workloads) {
    return llvm::all_of(workloads, [](auto workload) {
        const auto wlSizes = workload.getConstOutputSizes();
        const auto ch = wlSizes[Dims4D::Act::C.ind()];
        return ch == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16 ||
               ch == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32;
    });
}

bool doesWorkloadSupportSmallKernelOpt(const int64_t KX, const int64_t SX, ArrayRef<int64_t> workloadOutSz,
                                       bool isFp16Input) {
    // L1Opt can be enabled when kernelX = 3 and strideX = 1
    if (KX != 3 || SX != 1) {
        return false;
    }

    // Float16 input align to 16 generates performance regression.
    return isFp16Input ? workloadOutSz[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32
                       : workloadOutSz[Dims4D::Act::C.ind()] % VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16 == 0;
}

}  // namespace vpux::VPU::arch40xx
