//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/utils/permute_dma.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch37xx {
#define GEN_PASS_DECL_UNROLLPERMUTEDMA
#define GEN_PASS_DEF_UNROLLPERMUTEDMA
#include "vpux/compiler/NPU37XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch37xx

using namespace vpux;

namespace {

//
// PermuteDMARewriter
//

class PermuteDMARewriter final : public mlir::OpRewritePattern<VPUIP::PermuteDMAOp> {
public:
    PermuteDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::PermuteDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
        setDebugName("PermuteDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _dmaPortCount;
    Logger _log;
};

mlir::LogicalResult PermuteDMARewriter::matchAndRewrite(VPUIP::PermuteDMAOp permuteOp,
                                                        mlir::PatternRewriter& rewriter) const {
    return arch37xx::unrollPermuteDMA<arch37xx::UnrollSingleClusterPermuteDMA, arch37xx::UnrollMultiClusterPermuteDMA>(
            permuteOp, rewriter, _dmaPortCount, _log);
}

//
// UnrollPermuteDMAPass
//

class UnrollPermuteDMAPass final : public VPUIP::arch37xx::impl::UnrollPermuteDMABase<UnrollPermuteDMAPass> {
public:
    explicit UnrollPermuteDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollPermuteDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollPermuteDMAPass)) {
        return;
    }
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteDMARewriter>(&ctx, dmaPortCount, _log.nest());
    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollPermuteDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch37xx::createUnrollPermuteDMAPass(Logger log) {
    return std::make_unique<UnrollPermuteDMAPass>(log);
}
