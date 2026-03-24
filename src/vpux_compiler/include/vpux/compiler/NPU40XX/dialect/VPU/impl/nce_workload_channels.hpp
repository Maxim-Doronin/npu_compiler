//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Types.h>

using namespace vpux;

namespace vpux::VPU::arch40xx {

bool hasAnyChannelSupportedByKernelOptimization(ArrayRef<int64_t> supportedChannels, int64_t KX, int64_t SX);
SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> workloadsChannels, int64_t maxSlotsSum);
bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp);

}  // namespace vpux::VPU::arch40xx
