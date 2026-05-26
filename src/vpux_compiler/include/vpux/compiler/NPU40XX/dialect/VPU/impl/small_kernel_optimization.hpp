//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"

namespace vpux::VPU::arch40xx {

bool isSmallKernelOptimizationSupported(mlir::Operation* op, ArrayRef<VPU::DPUWorkloadOp> workloads);
bool doesWorkloadSupportSmallKernelOpt(int64_t KX, int64_t SX, ArrayRef<int64_t> workloadOutSz, bool isFp16Input);

}  // namespace vpux::VPU::arch40xx
