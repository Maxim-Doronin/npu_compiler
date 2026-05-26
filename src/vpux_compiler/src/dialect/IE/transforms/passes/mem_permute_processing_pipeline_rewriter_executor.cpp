//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/mem_permute_processing_pipeline_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
#define GEN_PASS_DECL_MEMPERMUTEPROCESSINGPIPELINEREWRITEREXECUTOR
#define GEN_PASS_DEF_MEMPERMUTEPROCESSINGPIPELINEREWRITEREXECUTOR
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// MemPermuteProcessingPipelineRewriterExecutorPass
//

class MemPermuteProcessingPipelineRewriterExecutorPass final :
        public impl::MemPermuteProcessingPipelineRewriterExecutorBase<MemPermuteProcessingPipelineRewriterExecutorPass>,
        public RewriterExecutorInterface {
public:
    using Base =
            impl::MemPermuteProcessingPipelineRewriterExecutorBase<MemPermuteProcessingPipelineRewriterExecutorPass>;

    explicit MemPermuteProcessingPipelineRewriterExecutorPass(const IE::ExpandActivationChannelsOptions& options,
                                                              Logger log)
            : _enableAdjustConvShapePass(options.enableAdjustConvShapePass) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnFunc() final;
    bool _enableAdjustConvShapePass;
};

mlir::LogicalResult MemPermuteProcessingPipelineRewriterExecutorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (rewriterName.hasValue()) {
        setRewriterName(rewriterName.getValue());
    }

    return mlir::success();
}

void MemPermuteProcessingPipelineRewriterExecutorPass::safeRunOnFunc() {
    const auto func = getOperation();
    auto& ctx = getContext();
    auto moduleOp = getModuleOp(func);

    auto strategy = IE::createMemPermuteProcessingPipelineStrategy(
            func,
            config::hasEnableSEPtrsOperations(moduleOp) || config::hasEnableExperimentalSEPtrsOperations(moduleOp),
            _enableAdjustConvShapePass);
    auto customRegistry = vpux::RegistryManager::createCustomRegistry();
    strategy->registerRewriters(*customRegistry, _log);

    if (mlir::failed(this->executeRewriters(&ctx, _log, func, customRegistry.get()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createMemPermuteProcessingPipelineRewriterExecutorPass(
        const IE::ExpandActivationChannelsOptions& options, Logger log) {
    return std::make_unique<MemPermuteProcessingPipelineRewriterExecutorPass>(options, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createMemPermuteProcessingPipelineRewriterExecutorPass(Logger log) {
    return createMemPermuteProcessingPipelineRewriterExecutorPass(IE::ExpandActivationChannelsOptions{}, log);
}
