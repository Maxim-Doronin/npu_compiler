//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/merge_vf_region_base_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU::VF::v2 {

//
// MergeVFRegionRewriter
//

class MergeVFRegionRewriter final : public MergeVFRegionBaseRewriter<VFCase> {
public:
    MergeVFRegionRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining, bool enablePrefetchTiling,
                          const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : MergeVFRegionBaseRewriter<VFCase>(ctx, enableVerticalFusionPipelining, enablePrefetchTiling, costFunction,
                                                log) {
    }

protected:
    bool canMergeVFOpsWithoutCostCheck(VFCase& mergedCase) const override;
    bool canSkipMergeVF(VFConfig& vfConfig, bool opsNeedTiling) const override;
    VPU::StrategyCost extractVFCost(VFConfig& vfConfig) const override;
    std::optional<int64_t> getOptimalTilingStrategy(const IVFSchedulingPtr& scheduling, const Dim dim,
                                                    const int64_t minTiles, int64_t& maxTiles,
                                                    TilingOperationStorage::UPtr& minStorage,
                                                    TilingOperationStorage::UPtr& maxStorage,
                                                    VFConfig& config) const override;
    bool cmxSizeExceedForEltwiseOpWithSwOpUser(VFConfig& currentConfig, ArrayRef<mlir::Operation*> parents,
                                               Logger log) const;
    std::deque<IVFSchedulingPtr> getVFSchedulingChecks(VFConfig& config) const override;
    std::shared_ptr<IVFScheduling<VFConfig>> detectScenario(VFConfig& vfConfig) const override;
    std::optional<VFCase> findVFTiling(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp prevOp,
                                       VPU::VerticalFusionOp currentOp) const override;
};

}  // namespace vpux::VPU::VF::v2
