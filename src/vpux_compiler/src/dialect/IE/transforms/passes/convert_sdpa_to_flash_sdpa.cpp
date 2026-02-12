//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSDPATOFLASHSDPA
#define GEN_PASS_DEF_CONVERTSDPATOFLASHSDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class SDPARewrite final : public mlir::OpRewritePattern<IE::SDPAOp> {
public:
    SDPARewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SDPAOp>(ctx), _log(log) {
        setDebugName("SDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SDPARewrite::matchAndRewrite(IE::SDPAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ctx = getContext();

    auto valueShape = getShape(origOp.getInputV());
    auto valueRank = valueShape.size();
    if (valueRank < 2) {
        return errorAt(origOp, "Invalid Value tensor rank '{0}'", valueRank);
    }

    auto queryShape = getShape(origOp.getInputQ());
    auto targetSeqLen = *(queryShape.end() - 2);

    auto vEmbedding = *(valueShape.end() - 1);

    // If scale input is not present, Query is scaled by a constant instead
    auto scale = mlir::Value{origOp.getInputScale()};
    if (scale == nullptr) {
        auto queryShape = getShape(origOp.getInputQ());
        auto qkEmbedding = queryShape.back();
        auto scaleValue = 1.0f / sqrtf(checked_cast<float>(qkEmbedding));

        auto scaleType = mlir::RankedTensorType::get({1}, getFp16Type(getContext()));
        auto scaleData = SmallVector<float>{scaleValue};
        auto scaleConstant =
                Const::createConst(rewriter, appendLoc(origOp.getLoc(), "scale_const"), scaleType, ArrayRef(scaleData));

        scale = scaleConstant;
    }

    auto scaledQuery = rewriter.create<IE::MultiplyOp>(appendLoc(origOp.getInputQ().getLoc(), "query_scaled"),
                                                       origOp.getInputQ(), scale, IE::AutoBroadcastType::NUMPY,
                                                       /*postOp=*/nullptr, /*clamp=*/nullptr,
                                                       /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);

    auto queryBatches = Shape(queryShape.begin(), queryShape.end() - 2);

    auto runningOutputShape = queryBatches;
    runningOutputShape.push_back(targetSeqLen);
    runningOutputShape.push_back(vEmbedding);
    auto runningOutputType = mlir::RankedTensorType::get(runningOutputShape.raw(), mlir::Float16Type::get(ctx));
    auto initRunningOutput = vpux::Const::createDenseConst(rewriter, appendLoc(origOp->getLoc(), "running_output"),
                                                           runningOutputType, 0.0f);

    auto runningMaxAndSumShape = std::move(queryBatches);
    runningMaxAndSumShape.push_back(targetSeqLen);
    auto runningMaxType = mlir::RankedTensorType::get(runningMaxAndSumShape.raw(), mlir::Float16Type::get(ctx));
    auto initRunningMax = vpux::Const::createDenseConst(rewriter, appendLoc(origOp->getLoc(), "running_max"),
                                                        runningMaxType, -std::numeric_limits<float>::infinity());

    auto runningSumType = mlir::RankedTensorType::get(runningMaxAndSumShape.raw(), mlir::Float32Type::get(ctx));
    auto initRunningSum =
            vpux::Const::createDenseConst(rewriter, appendLoc(origOp->getLoc(), "running_sum"), runningSumType, 0.0f);

    auto trueAttr = mlir::BoolAttr::get(ctx, true);
    auto flashSdpa = rewriter.create<IE::FlashSDPAOp>(appendLoc(origOp->getLoc(), "FlashAttention"), scaledQuery,
                                                      origOp.getInputK(), origOp.getInputV(), initRunningOutput,
                                                      initRunningMax, initRunningSum, origOp.getInputMask(),
                                                      /*isHead*/ trueAttr, /*isTail*/ trueAttr,
                                                      /*sourceSeqLenPadSize*/ getIntAttr(ctx, 0));

    // Output type depends on the initial running output tensor which is initialized with a fp16 constant
    // Original IE::SDPA operation might have fp32 output type so we need to add a Convert operation
    auto newResult = flashSdpa.getResultRunningOutput();
    if (newResult.getType().getElementType() != origOp.getType().getElementType()) {
        auto convertToTarget = rewriter.create<IE::ConvertOp>(appendLoc(newResult.getLoc(), "convert"), newResult,
                                                              origOp.getType().getElementType());

        newResult = convertToTarget;
    }

    rewriter.replaceOp(origOp, newResult);

    return mlir::success();
}

//
// ConvertSDPAToFlashSDPA
//

class ConvertSDPAToFlashSDPA final : public IE::impl::ConvertSDPAToFlashSDPABase<ConvertSDPAToFlashSDPA> {
public:
    explicit ConvertSDPAToFlashSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertSDPAToFlashSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    const auto isLegal = [](IE::SDPAOp) {
        // A more sophisticated condition should be implemented to decide
        // when this conversion is necessary or favorable
        return false;
    };

    target.addDynamicallyLegalOp<IE::SDPAOp>(isLegal);
    target.addLegalOp<IE::FlashSDPAOp>();
    target.addLegalOp<IE::MultiplyOp>();
    target.addLegalOp<IE::ConvertOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSDPAToFlashSDPAPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSDPAToFlashSDPAPass(Logger log) {
    return std::make_unique<ConvertSDPAToFlashSDPA>(log);
}
