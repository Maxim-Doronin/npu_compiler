//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

namespace VPUNN {
class VPUCostModel;
}  // namespace VPUNN

namespace vpux::VPUIP {

int64_t computeSplitCost(mlir::MLIRContext* ctx, const WorkloadSplit& split, const WorkloadCostParams& params,
                         VPUNN::VPUCostModel& costModel, LogCb logCb = emptyLogCb);

}  // namespace vpux::VPUIP
