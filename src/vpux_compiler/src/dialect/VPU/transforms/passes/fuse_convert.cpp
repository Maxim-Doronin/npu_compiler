//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_FUSECONVERTPASS
#define GEN_PASS_DEF_FUSECONVERTPASS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// FuseD2sConvertRewrite
//

class FuseD2sConvertRewrite final : public mlir::OpRewritePattern<VPU::ConvertOp> {
public:
    FuseD2sConvertRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ConvertOp>(ctx), _log(log) {
        this->setDebugName("FuseD2sConvertRewrite");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseD2sConvertRewrite::matchAndRewrite(VPU::ConvertOp convertOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", convertOp->getName(), convertOp->getLoc());
    auto nestedLogger = _log.nest();

    auto d2sOp = convertOp.getInput().getDefiningOp<VPU::DepthToSpaceOp>();
    if (d2sOp == nullptr) {
        nestedLogger.trace("ConvertOp does not have DepthToSpaceOp input {0}", convertOp->getName());
        return mlir::failure();
    }

    if (d2sOp.getDstElemType().has_value()) {
        nestedLogger.trace("DepthToSpaceOp already has dstElemType attr set");
        return mlir::failure();
    }
    if (!d2sOp.getResult().hasOneUse()) {
        nestedLogger.trace("DepthToSpaceOp has multiple users");
        return mlir::failure();
    }

    const auto mode = d2sOp.getMode();
    const auto blockSize = d2sOp.getBlockSize();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getInput().getType());
    const auto inShape = inType.getShape();
    const auto inElemType = inType.getElementType();
    const auto inCh = inShape[Dims4D::Act::C];
    const auto padding = d2sOp.getPaddedChannels();

    auto supportedConfig = (mode == IE::DepthToSpaceMode::BLOCKS_FIRST) && (blockSize == 2) && (inCh == 16) &&
                           (!padding.has_value()) && (inElemType.isF16()) && (convertOp.getDstElemType().isF32());

    if (supportedConfig) {
        auto origD2sOutType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getOutput().getType());
        auto newOp = mlir::dyn_cast_or_null<VPU::DepthToSpaceOp>(rewriter.clone(*d2sOp));
        newOp.setDstElemType(convertOp.getDstElemType());
        newOp.getOutput().setType(
                mlir::cast<mlir::RankedTensorType>(origD2sOutType.changeElemType(convertOp.getDstElemType())));
        rewriter.replaceOp(convertOp, newOp->getResult(0));
        return mlir::success();
    }

    nestedLogger.trace("Unsupported DepthToSpaceOp/ConvertOp config");
    return mlir::failure();
}

//
// FuseConvertPass
//

class FuseConvertPass final : public VPU::impl::FuseConvertPassBase<FuseConvertPass> {
public:
    explicit FuseConvertPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseConvertPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseD2sConvertRewrite>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseConvertPass(Logger log) {
    return std::make_unique<FuseConvertPass>(log);
}
