//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

#include <vector>

namespace vpux {

using VariantInfoArray = llvm::SmallVector<profiling::DPUVariantInfo>;

VariantInfoArray extractVariantInfoFromOp(VPUIP::NCEClusterTaskOp op);

profiling::TensorInfo extractTensorInfoFromOp(VPUIP::NCEClusterTaskOp op);

profiling::TensorInfo extractTensorInfoFromOp(VPUIP::SwKernelOp op);

std::pair<profiling::TensorInfo, profiling::TensorInfo> extractTensorInfoFromOp(VPUIP::DMATypeOpInterface op);

}  // namespace vpux
