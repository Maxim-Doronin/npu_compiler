//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/batch_op_processing_pipeline_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
#define GEN_PASS_DECL_BATCHOPPROCESSINGPIPELINEREWRITEREXECUTOR
#define GEN_PASS_DEF_BATCHOPPROCESSINGPIPELINEREWRITEREXECUTOR
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// BatchOpProcessingPipelineRewriterExecutorPass
//

class BatchOpProcessingPipelineRewriterExecutorPass final :
        public impl::BatchOpProcessingPipelineRewriterExecutorBase<BatchOpProcessingPipelineRewriterExecutorPass>,
        public RewriterExecutorInterface {
public:
    using Base = impl::BatchOpProcessingPipelineRewriterExecutorBase<BatchOpProcessingPipelineRewriterExecutorPass>;

    explicit BatchOpProcessingPipelineRewriterExecutorPass(const IE::TransformOptions& options, Logger log)
            : _enableGroupedMatMul(options.enableGroupedMatMul) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnFunc() final;
    bool _enableGroupedMatMul = false;
};

mlir::LogicalResult BatchOpProcessingPipelineRewriterExecutorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (rewriterName.hasValue()) {
        setRewriterName(rewriterName.getValue());
    }

    if (enableGroupedMatMul.hasValue()) {
        _enableGroupedMatMul = enableGroupedMatMul.getValue();
    }

    return mlir::success();
}

void BatchOpProcessingPipelineRewriterExecutorPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    auto strategy = IE::createBatchOpProcessingPipelineStrategy(func, _enableGroupedMatMul);
    auto customRegistry = vpux::RegistryManager::createCustomRegistry();
    strategy->registerRewriters(*customRegistry, _log);

    if (mlir::failed(this->executeRewriters(&ctx, _log, func, customRegistry.get()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createBatchOpProcessingPipelineRewriterExecutorPass(
        const IE::TransformOptions& options, Logger log) {
    return std::make_unique<BatchOpProcessingPipelineRewriterExecutorPass>(options, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createBatchOpProcessingPipelineRewriterExecutorPass(Logger log) {
    IE::TransformOptions options;
    return std::make_unique<BatchOpProcessingPipelineRewriterExecutorPass>(options, log);
}
