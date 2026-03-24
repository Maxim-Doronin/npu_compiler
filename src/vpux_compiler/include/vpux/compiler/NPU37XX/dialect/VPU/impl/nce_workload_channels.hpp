//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Types.h>

using namespace vpux;

namespace vpux::VPU::arch37xx {

bool hasAnyChannelSupportedByKernelOptimization();
SmallVector<int64_t> getChannelsSupportedByKernelOptimization();
bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp);

}  // namespace vpux::VPU::arch37xx
