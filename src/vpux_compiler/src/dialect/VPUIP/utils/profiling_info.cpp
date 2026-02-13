//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/profiling_info.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

mlir::Type VPUIP::getTimestampType(mlir::MLIRContext* ctx) {
    return getUInt64Type(ctx);
}

void VPUIP::setWorkloadIds(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    int32_t workloadId = 0;
    int32_t prevClusterId = -1;
    nceClusterTaskOp.walk([&](VPUIP::DPUTaskOp dpuTaskOp) {
        if (dpuTaskOp.getClusterId().has_value()) {
            int32_t clusterId = checked_cast<int32_t>(dpuTaskOp.getClusterId().value());
            if (prevClusterId != clusterId) {
                workloadId = 0;
            }
            prevClusterId = clusterId;
        }
        dpuTaskOp.setWorkloadIdAttr(vpux::getIntAttr(dpuTaskOp->getContext(), workloadId));
        ++workloadId;
    });
}
