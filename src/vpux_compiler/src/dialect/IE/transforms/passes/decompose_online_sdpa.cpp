//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEONLINESDPA
#define GEN_PASS_DEF_DECOMPOSEONLINESDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class OnlineSDPARewrite final : public mlir::OpRewritePattern<IE::OnlineSDPAOp> {
public:
    OnlineSDPARewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::OnlineSDPAOp>(ctx), _log(log) {
        setDebugName("OnlineSDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::OnlineSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

Const::DeclareOp createDenseConstant(mlir::PatternRewriter& rewriter, ShapeRef shape, float value) {
    const auto runningMaxType = mlir::RankedTensorType::get(shape.raw(), rewriter.getF16Type());
    const auto values = SmallVector<type::float16>(shape.totalSize(), type::float16(value));
    auto content = Const::ContentAttr::get(Const::createConstContent(runningMaxType, ArrayRef<type::float16>(values)));

    return rewriter.create<Const::DeclareOp>(mlir::UnknownLoc::get(rewriter.getContext()), content.getType(), content);
}

mlir::LogicalResult OnlineSDPARewrite::matchAndRewrite(IE::OnlineSDPAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto queryShape = getShape(origOp.getQuery());
    // Tensors for Max and Sum tensors reduces the last dimension.
    auto bufferShape = Shape(queryShape);
    bufferShape.pop_back();

    auto initialRunningMax = createDenseConstant(rewriter, bufferShape, -std::numeric_limits<float>::infinity());
    auto initialRunningSum = createDenseConstant(rewriter, bufferShape, 0.0f);
    auto initialPartialOutput = createDenseConstant(rewriter, queryShape, 0.0f);

    auto incrementalSDPA = [&]() {
        const auto numInputs = origOp.getInputs().size();
        const auto loc = appendLoc(origOp->getLoc(), "incremental");

        // TODO: IsCasual attribute
        const auto casual = nullptr;

        if (numInputs == 3) {
            return rewriter.create<IE::IncrementalSDPAOp>(loc, origOp.getQuery(), origOp.getKey(), origOp.getValue(),
                                                          initialRunningMax, initialRunningSum, initialPartialOutput,
                                                          nullptr, nullptr, casual, origOp.getKvNumBlocksAttr());
        } else if (numInputs == 4) {
            return rewriter.create<IE::IncrementalSDPAOp>(
                    loc, origOp.getQuery(), origOp.getKey(), origOp.getValue(), initialRunningMax, initialRunningSum,
                    initialPartialOutput, origOp.getAttentionMask(), nullptr, casual, origOp.getKvNumBlocksAttr());
        } else if (numInputs == 5) {
            return rewriter.create<IE::IncrementalSDPAOp>(loc, origOp.getQuery(), origOp.getKey(), origOp.getValue(),
                                                          initialRunningMax, initialRunningSum, initialPartialOutput,
                                                          origOp.getAttentionMask(), origOp.getScale(), casual,
                                                          origOp.getKvNumBlocksAttr());
        } else {
            VPUX_THROW("{0} has unexpected number of inputs: {1}", origOp->getName(), numInputs);
        }
    }();

    auto bufferRank = static_cast<int64_t>(bufferShape.size());
    auto lastDimension = getIntArrayAttr(rewriter.getContext(), SmallVector<int64_t>{bufferRank});
    auto unsqueezedRunningSum = rewriter.create<IE::UnsqueezeOp>(
            takeOpLoc(origOp, "unsqueezed_running_sum"), incrementalSDPA.getResultRunningSum(), nullptr, lastDimension);

    auto broadcastNumpy = IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY);
    auto finalDivision =
            rewriter.create<IE::DivideOp>(takeOpLoc(origOp, "divide"), incrementalSDPA.getResultPartialOutput(),
                                          unsqueezedRunningSum, broadcastNumpy);

    rewriter.replaceOp(origOp, finalDivision);

    return mlir::success();
}

//
// DecomposeOnlineSDPA
//

class DecomposeOnlineSDPA final : public IE::impl::DecomposeOnlineSDPABase<DecomposeOnlineSDPA> {
public:
    explicit DecomposeOnlineSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void DecomposeOnlineSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::OnlineSDPAOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::IncrementalSDPAOp>();
    target.addLegalOp<IE::DivideOp>();
    target.addLegalOp<IE::UnsqueezeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OnlineSDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeOnlineSDPAPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeOnlineSDPAPass(Logger log) {
    return std::make_unique<DecomposeOnlineSDPA>(log);
}
