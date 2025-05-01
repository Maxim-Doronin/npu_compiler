//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"

namespace vpux {

mlir::LogicalResult genericNCEWorkloadSplit(VPU::NCEOpInterface nceOp, mlir::PatternRewriter& rewriter,
                                            VPU::ArchKind arch, int64_t numDPU,
                                            std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log);

VPU::DistributedTensorType getDistributedTensor(const mlir::Value value);

/**
 * Map distributedType mode to VPUNN:ISIStrategy and outputWriteTiles
 *     DistributionMode         ISIStrategy      OutputWriteTiles
 * SEGMENTED|MULTICASTED        CLUSTERING          numTiles
 * SEGMENTED|DUPLICATED         SPLIT_OVER_K        numTiles
 *      else                    CLUSTERING              1
 */
template <typename TensorType, typename = std::enable_if_t<std::is_same_v<VPU::DistributedTensorType, TensorType> ||
                                                           std::is_same_v<VPUIP::DistributedBufferType, TensorType>>>
VPUNN::ISIStrategy getISIStrategyForType(TensorType type, unsigned int& outputWriteTiles) {
    const auto distributionAttr = type.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    auto numClusters = distributionAttr.getNumClusters().getInt();

    if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        outputWriteTiles = numClusters;
        return VPUNN::ISIStrategy::CLUSTERING;
    }
    if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        outputWriteTiles = numClusters;
        return VPUNN::ISIStrategy::SPLIT_OVER_K;
    }
    return VPUNN::ISIStrategy::CLUSTERING;
}

}  // namespace vpux
