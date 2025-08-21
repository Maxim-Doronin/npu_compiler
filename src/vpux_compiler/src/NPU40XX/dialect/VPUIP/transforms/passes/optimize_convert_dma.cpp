//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_OPTIMIZECONVERTDMAOP
#define GEN_PASS_DEF_OPTIMIZECONVERTDMAOP
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

VPUIP::LayerOpInterface getConvertDMAOp(mlir::Operation* maybeConvertDMAOperation) {
    if (auto convertDMAOp = mlir::dyn_cast_or_null<VPUIP::ConvertDMAOp>(maybeConvertDMAOperation)) {
        return mlir::cast<VPUIP::LayerOpInterface>(*convertDMAOp);
    }
    return nullptr;
}

VPUIP::LayerOpInterface getCopyOp(mlir::Operation* sourceOp) {
    return mlir::dyn_cast_or_null<VPUIP::CopyOp>(sourceOp);
}

void replaceOpWithNewConvertDMAOp(mlir::PatternRewriter& rewriter, mlir::Value input, mlir::Value outputBuff,
                                  mlir::Operation* opToReplace) {
    rewriter.replaceOpWithNewOp<VPUIP::ConvertDMAOp>(opToReplace, input, outputBuff);
}

class ConvertDMACopy : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    ConvertDMACopy(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertDMACopy::matchAndRewrite(VPUIP::CopyOp copy, mlir::PatternRewriter& rewriter) const {
    _log.trace("ConvertDMACopy: Copy at {0}", copy->getLoc());
    auto nestedLogger = _log.nest();

    auto copyOp = getCopyOp(copy);
    if (copyOp == nullptr) {
        nestedLogger.trace("Couldn't find the copyOp");
        return mlir::failure();
    }

    auto copyInput = copyOp->getOperand(0);
    auto convertDMAOp = getConvertDMAOp(copyInput.getDefiningOp());
    if (convertDMAOp == nullptr) {
        nestedLogger.trace("Input ConvertDMAOp not found {0}", copyInput.getLoc());
        return mlir::failure();
    }

    if (!convertDMAOp->hasOneUse()) {
        nestedLogger.trace("ConvertDMA has multiple use {0}", copyOp.getLoc());
        return mlir::failure();
    }

    auto newConvertDMAInput = convertDMAOp->getOperand(0);
    auto parentCopy = copyOp.getOperation();
    auto outputBuff = copyOp.getOutputs()[0];

    auto newConvertDMAInputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(newConvertDMAInput.getType());
    auto newConvertDMAOutputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputBuff.getType());
    if (newConvertDMAInputDistType != nullptr && newConvertDMAOutputDistType != nullptr &&
        mlir::failed(VPU::areDistributionAttrsCompatible(newConvertDMAInputDistType, newConvertDMAOutputDistType,
                                                         /*allowDifferentPerClusterMemoryView = */ false))) {
        nestedLogger.trace("ConvertDMA will have incompatible input and output distributions after fused with copy",
                           copyOp.getLoc());
        return mlir::failure();
    }

    // Temporarily disable fuse of ClusterConvertDMA(from SEGMENDTED) and Copy(toDDR) due to wrong DMA descriptors
    // generated for this case
    // Tracked in: E#101270
    if (newConvertDMAInputDistType != nullptr && newConvertDMAOutputDistType == nullptr) {
        const auto inDistMode = newConvertDMAInputDistType.getDistribution().getMode().getValue();
        const auto outMemKind = mlir::cast<vpux::NDTypeInterface>(outputBuff.getType()).getMemoryKind();
        if (inDistMode == VPU::DistributionMode::SEGMENTED && outMemKind == VPU::MemoryKind::DDR) {
            return mlir::failure();
        }
    }

    rewriter.setInsertionPointAfter(parentCopy);

    replaceOpWithNewConvertDMAOp(rewriter, newConvertDMAInput, outputBuff, parentCopy);

    if (convertDMAOp->use_empty()) {
        rewriter.eraseOp(convertDMAOp);
    }
    nestedLogger.trace("Successfully optimized ConvertDMA->Copy pattern");
    return mlir::success();
}

class CopyConvertDMA : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyConvertDMA(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
        setDebugName("CopyConvertDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyConvertDMA::matchAndRewrite(VPUIP::CopyOp copy, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Copy at {0}", getDebugName(), copy->getLoc());
    auto nestedLogger = _log.nest();

    auto copyOp = getCopyOp(copy);
    if (copyOp == nullptr) {
        nestedLogger.trace("Couldn't find the copyOp");
        return mlir::failure();
    }

    // Copy op should have only one result
    if (copyOp->getResults().size() != 1) {
        nestedLogger.trace("Copy op should have only one result {0}", copyOp.getLoc());
        return mlir::failure();
    }

    if (!copyOp->hasOneUse()) {
        nestedLogger.trace("Copy op has multiple use {0}", copyOp.getLoc());
        return mlir::failure();
    }

    auto copyOutput = *copyOp->getResult(0).getUsers().begin();
    auto convertDMAOp = getConvertDMAOp(copyOutput);
    if (convertDMAOp == nullptr) {
        nestedLogger.trace("Result ConvertDMAOp not found {0}", copyOutput->getLoc());
        return mlir::failure();
    }

    auto newConvertDMAInput = copyOp->getOperand(0);
    auto outputBuff = convertDMAOp.getOutputs()[0];

    auto newConvertDMAInputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(newConvertDMAInput.getType());
    auto newConvertDMAOutputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputBuff.getType());
    if (newConvertDMAInputDistType != nullptr && newConvertDMAOutputDistType != nullptr &&
        mlir::failed(VPU::areDistributionAttrsCompatible(newConvertDMAInputDistType, newConvertDMAOutputDistType,
                                                         /*allowDifferentPerClusterMemoryView = */ false))) {
        nestedLogger.trace("ConvertDMA will have incompatible input and output distributions after fused with copy",
                           copyOp.getLoc());
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(convertDMAOp.getOperation());

    replaceOpWithNewConvertDMAOp(rewriter, newConvertDMAInput, outputBuff, convertDMAOp);

    if (copyOp->use_empty()) {
        rewriter.eraseOp(copyOp);
    }
    nestedLogger.trace("Successfully optimized Copy->ClusterConvertDMA pattern");
    return mlir::success();
}

//
// OptimizeConvertDMAPass
//

class OptimizeConvertDMAPass final : public VPUIP::arch40xx::impl::OptimizeConvertDMAOpBase<OptimizeConvertDMAPass> {
public:
    explicit OptimizeConvertDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeConvertDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertDMACopy>(&ctx, _log);
    patterns.add<CopyConvertDMA>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConvertDMAOpPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createOptimizeConvertDMAOpPass(Logger log) {
    return std::make_unique<OptimizeConvertDMAPass>(log);
}
