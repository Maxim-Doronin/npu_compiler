//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSDPATOONLINESDPA
#define GEN_PASS_DEF_CONVERTSDPATOONLINESDPA
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

    auto onlineSdpa = rewriter.create<IE::OnlineSDPAOp>(appendLoc(origOp->getLoc(), "onlineSDPA"), origOp.getInputQ(),
                                                        origOp.getInputK(), origOp.getInputV(), origOp.getInputMask(),
                                                        /* scale */ nullptr, /* kvNumBlocks */ nullptr);

    rewriter.replaceOp(origOp, onlineSdpa.getOutput());

    return mlir::success();
}

//
// ConvertSDPAToOnlineSDPA
//

class ConvertSDPAToOnlineSDPA final : public IE::impl::ConvertSDPAToOnlineSDPABase<ConvertSDPAToOnlineSDPA> {
public:
    explicit ConvertSDPAToOnlineSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertSDPAToOnlineSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    const auto isLegal = [](IE::SDPAOp) {
        // A more sophisticated condition should be implemented to decide
        // when this conversion is necessary or favorable
        return false;
    };

    target.addDynamicallyLegalOp<IE::SDPAOp>(isLegal);
    target.addLegalOp<IE::OnlineSDPAOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSDPAToOnlineSDPAPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createConvertSDPAToOnlineSDPAPass(Logger log) {
    return std::make_unique<ConvertSDPAToOnlineSDPA>(log);
}
