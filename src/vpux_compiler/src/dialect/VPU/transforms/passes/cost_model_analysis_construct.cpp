//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_COSTMODELANALYSISCONSTRUCT
#define GEN_PASS_DEF_COSTMODELANALYSISCONSTRUCT
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// CostModelAnalysisConstructPass
//
class CostModelAnalysisConstructPass final :
        public VPU::impl::CostModelAnalysisConstructBase<CostModelAnalysisConstructPass> {
public:
    explicit CostModelAnalysisConstructPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void CostModelAnalysisConstructPass::safeRunOnModule() {
    auto maybeCostModelAnalysis = getCachedAnalysis<VPU::CostModelAnalysis>();
    if (maybeCostModelAnalysis.has_value()) {
        _log.trace("CostModelAnalysis already preserved");
    } else {
        std::ignore = getAnalysis<vpux::VPU::CostModelAnalysis>();
        _log.trace("Created and preserved costModelAnalysis");
    }
    auto maybeLayerCostModelAnalysis = getCachedAnalysis<VPU::LayerCostModelAnalysis>();
    if (maybeLayerCostModelAnalysis.has_value()) {
        _log.trace("LayerCostModelAnalysis already preserved");
    } else {
        std::ignore = getAnalysis<vpux::VPU::LayerCostModelAnalysis>();
        _log.trace("Created and preserved layerCostModelAnalysis");
    }
}

}  // namespace

//
// createCostModelAnalysisConstruct
//

std::unique_ptr<mlir::Pass> VPU::createCostModelAnalysisConstructPass(Logger log) {
    return std::make_unique<CostModelAnalysisConstructPass>(log);
}
