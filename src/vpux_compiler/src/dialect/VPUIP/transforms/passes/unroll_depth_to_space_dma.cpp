//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLDEPTHTOSPACEDMA
#define GEN_PASS_DEF_UNROLLDEPTHTOSPACEDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UnrollDepthToSpaceDMAPass
//

class UnrollDepthToSpaceDMAPass final : public VPUIP::impl::UnrollDepthToSpaceDMABase<UnrollDepthToSpaceDMAPass> {
public:
    explicit UnrollDepthToSpaceDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollDepthToSpaceDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollDepthToSpaceDMAPass)) {
        return;
    }

    SmallVector<mlir::RewritePatternSet> patternSets;

    auto& strategyFactory = VPUIP::getVPUIPStrategyFactory(&ctx);
    auto dmaPortCount = config::getNumOfDMAPorts(func);
    auto unrollStrategy = strategyFactory->getUnrollDepthToSpaceDMAStrategy(&ctx, dmaPortCount);
    unrollStrategy->addPatterns(patternSets, _log);

    for (auto& patternSet : patternSets) {
        collectOpsAndApplyPatterns(func, std::move(patternSet));
    }
}

}  // namespace

//
// createUnrollDepthToSpaceDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollDepthToSpaceDMAPass(Logger log) {
    return std::make_unique<UnrollDepthToSpaceDMAPass>(log);
}
