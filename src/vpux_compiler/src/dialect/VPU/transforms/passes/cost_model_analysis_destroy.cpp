//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_COSTMODELANALYSISDESTROY
#define GEN_PASS_DEF_COSTMODELANALYSISDESTROY
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// CostModelAnalysisDestroyPass
//
class CostModelAnalysisDestroyPass final :
        public VPU::impl::CostModelAnalysisDestroyBase<CostModelAnalysisDestroyPass> {
public:
    explicit CostModelAnalysisDestroyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void CostModelAnalysisDestroyPass::safeRunOnModule() {
    auto maybeCostModelAnalysis = getCachedAnalysis<VPU::CostModelAnalysis>();
    if (maybeCostModelAnalysis.has_value()) {
        auto& costModelAnalysis = maybeCostModelAnalysis.value().get();
        costModelAnalysis.invalidate();
        _log.trace("Cost model analysis is destroyed");
    }
    auto maybeLayerCostModelAnalysis = getCachedAnalysis<VPU::LayerCostModelAnalysis>();
    if (maybeLayerCostModelAnalysis.has_value()) {
        auto& layerCostModelAnalysis = maybeLayerCostModelAnalysis.value().get();
        layerCostModelAnalysis.invalidate();
        _log.trace("Layer cost model analysis is destroyed");
    }
}

}  // namespace

//
// createCostModelAnalysisDestroy
//

std::unique_ptr<mlir::Pass> VPU::createCostModelAnalysisDestroyPass(Logger log) {
    return std::make_unique<CostModelAnalysisDestroyPass>(log);
}
