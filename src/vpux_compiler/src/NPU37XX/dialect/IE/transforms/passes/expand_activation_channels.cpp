//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/expand_activation_channels.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE::arch37xx {
#define GEN_PASS_DECL_EXPANDACTIVATIONCHANNELS
#define GEN_PASS_DEF_EXPANDACTIVATIONCHANNELS
#include "vpux/compiler/NPU37XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch37xx

using namespace vpux;

namespace {

//
// ExpandActivationChannelsPass
//

class ExpandActivationChannelsPass final :
        public IE::arch37xx::impl::ExpandActivationChannelsBase<ExpandActivationChannelsPass> {
public:
    explicit ExpandActivationChannelsPass(const bool seOpsEnabled, Logger log): _seOpsEnabled(seOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
};

mlir::LogicalResult ExpandActivationChannelsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }

    return mlir::success();
}

void ExpandActivationChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegal = [&](mlir::Operation* op) {
        if (!_seOpsEnabled && mlir::isa<IE::SEOpInterface>(op)) {
            return true;
        }

        if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            return iface.verifyChannels().succeeded();
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal(isLegal);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ExpandOp, IE::SliceOp>();
    target.addLegalOp<IE::MultiplyOp, IE::SubtractOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::MaxPoolRewriter>(&ctx, _log);
    patterns.add<IE::AvgPoolRewriter>(&ctx, _log);
    patterns.add<IE::EltwiseRewriter<IE::AddOp>>(&ctx, _log);
    patterns.add<IE::ConvolutionRewriter>(&ctx, _log);
    patterns.add<IE::GroupConvolutionRewriter>(&ctx, _log);
    patterns.add<IE::MatMulRewriter>(&ctx, _log);
    patterns.add<IE::SoftMaxRewriter>(&ctx, _log);

    if (_seOpsEnabled) {
        patterns.add<IE::InterpolateRewriter>(&ctx, _log);
        patterns.add<IE::TransposedConvolutionRewriter>(&ctx, _log);
        patterns.add<IE::PadRewriter>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createExpandActivationChannelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createExpandActivationChannelsPass(const bool seOpsEnabled,
                                                                                   Logger log) {
    return std::make_unique<ExpandActivationChannelsPass>(seOpsEnabled, log);
}
