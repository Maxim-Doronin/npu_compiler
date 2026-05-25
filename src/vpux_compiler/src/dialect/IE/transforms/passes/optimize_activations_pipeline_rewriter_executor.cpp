//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/optimize_activations_pipeline_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
#define GEN_PASS_DECL_OPTIMIZEACTIVATIONSPIPELINEREWRITEREXECUTOR
#define GEN_PASS_DEF_OPTIMIZEACTIVATIONSPIPELINEREWRITEREXECUTOR
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// OptimizeActivationsPipelineRewriterExecutorPass
//

class OptimizeActivationsPipelineRewriterExecutorPass final :
        public impl::OptimizeActivationsPipelineRewriterExecutorBase<OptimizeActivationsPipelineRewriterExecutorPass>,
        public RewriterExecutorInterface {
public:
    using Base = impl::OptimizeActivationsPipelineRewriterExecutorBase<OptimizeActivationsPipelineRewriterExecutorPass>;

    explicit OptimizeActivationsPipelineRewriterExecutorPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnFunc() final;
};

mlir::LogicalResult OptimizeActivationsPipelineRewriterExecutorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (rewriterName.hasValue()) {
        setRewriterName(rewriterName.getValue());
    }

    return mlir::success();
}

void OptimizeActivationsPipelineRewriterExecutorPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    const auto moduleOp = getModuleOp(func);

    auto strategy = IE::createOptimizeActivationsPipelineStrategy(
            func,
            config::hasEnableSEPtrsOperations(moduleOp) || config::hasEnableExperimentalSEPtrsOperations(moduleOp));
    auto customRegistry = vpux::RegistryManager::createCustomRegistry();
    strategy->registerRewriters(*customRegistry, _log);

    if (mlir::failed(this->executeRewriters(&ctx, _log, func, customRegistry.get()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeActivationsPipelineRewriterExecutorPass(Logger log) {
    return std::make_unique<OptimizeActivationsPipelineRewriterExecutorPass>(log);
}
