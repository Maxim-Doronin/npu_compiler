//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/small_kernel_optimization.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"

using namespace vpux;

namespace vpux::VPU::arch50xx {

bool isSmallKernelOptimizationSupported(mlir::Operation* op, const int64_t KX, const int64_t KY,
                                        ArrayRef<VPU::DPUWorkloadOp> workloads) {
    // Errata E#94064: check conditions for NPU50XX, if true, optimization is disabled
    const auto forbiddenConfigurations = llvm::any_of(workloads, [](auto workload) {
        const auto wlSizes = workload.getConstOutputSizes();
        const auto padCountLeft = workload.getPadAttribute().getLeft().getInt();
        return wlSizes[Dims4D::Act::H.ind()] < 8 && wlSizes[Dims4D::Act::W.ind()] < 5 && padCountLeft == 1;
    });

    if (forbiddenConfigurations && (KY == 1 || KX == 1)) {
        return false;
    }

    return VPU::arch40xx::isSmallKernelOptimizationSupported(op, workloads);
}

bool doesWorkloadSupportSmallKernelOpt(const int64_t KX, const int64_t SX, ArrayRef<int64_t> workloadOutSz,
                                       bool isFp16Input, [[maybe_unused]] const int64_t KY,
                                       [[maybe_unused]] const int64_t padLeft) {
    const bool forbiddenConfiguration =
            workloadOutSz[Dims4D::Act::H.ind()] < 8 && workloadOutSz[Dims4D::Act::W.ind()] < 5 && padLeft == 1;

    if (forbiddenConfiguration && (KY == 1 || KX == 1)) {
        return false;
    }

    return VPU::arch40xx::doesWorkloadSupportSmallKernelOpt(KX, SX, workloadOutSz, isFp16Input);
}

}  // namespace vpux::VPU::arch50xx
