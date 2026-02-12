//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/minimal_vf_scheduling.hpp"

namespace vpux::VPU::VF::v2 {
MinimalRequirementsVFScheduling::MinimalRequirementsVFScheduling(Logger log, bool prefetching)
        : v2::VFScheduling(log, prefetching) {
}

bool MinimalRequirementsVFScheduling::validate(VFConfig& /*config*/, const TilingOperationStorage::UPtr& tilingInfo,
                                               const Byte /*reservedMemory*/) const {
    return tilingInfo != nullptr;
}

VFScenario MinimalRequirementsVFScheduling::getType() const {
    return VFScenario::MINIMAL;
}

bool MinimalRequirementsVFScheduling::isSharedWeightsSupported(VFConfig&) const {
    return false;
}

void MinimalRequirementsVFScheduling::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, SmallVector<StrategyCost>& prefetchCostList,
        const size_t index) const {
    const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();

    if (isInput) {
        return;
    }

    VPUX_THROW_WHEN(index >= prefetchCostList.size(), "Index {0} out of range for prefetchCostList of size {1}", index,
                    prefetchCostList.size());
    StrategyCost parentCost = getParentCost(operation, isolatedOperCost);
    reduceCostWithPrefetchedDMA(parentCost, prefetchCost, prefetchCostList[index]);

    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}
}  // namespace vpux::VPU::VF::v2
