//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_QDQOPTIMIZATIONAGGRESSIVE
#define GEN_PASS_DEF_QDQOPTIMIZATIONAGGRESSIVE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class QDQOptimizationAggressivePass final :
        public IE::impl::QDQOptimizationAggressiveBase<QDQOptimizationAggressivePass> {
public:
    explicit QDQOptimizationAggressivePass(const bool fuseFQAndMulWithNonConstInput, Logger log)
            : _log(log), _fuseFQAndMulWithNonConstInput(fuseFQAndMulWithNonConstInput) {
        Base::initLogger(_log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    Logger _log;
    bool _fuseFQAndMulWithNonConstInput;
};

}  // namespace

void QDQOptimizationAggressivePass::safeRunOnModule() {
    auto moduleOp = getOperation();

    auto enableQDQOptimizationAggressive = config::hasEnableQDQOptimizationAggressive(moduleOp);
    mlir::OpPassManager dynamicPM("builtin.module");
    if (enableQDQOptimizationAggressive) {
        dynamicPM.addNestedPass<mlir::func::FuncOp>(IE::createAdjustFakeQdqParamsPass(_log));
        dynamicPM.addNestedPass<mlir::func::FuncOp>(
                IE::createFuseQuantizationMultiplyPass(_fuseFQAndMulWithNonConstInput, _log));
        dynamicPM.addNestedPass<mlir::func::FuncOp>(IE::createHandleU16FakeQuantizePass(_log));
    }

    if (failed(runPipeline(dynamicPM, moduleOp))) {
        signalPassFailure();
    }
}

//
// createQDQOptimizationAggressivePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createQDQOptimizationAggressivePass(const bool fuseFQAndMulWithNonConstInput,
                                                                          Logger log) {
    return std::make_unique<QDQOptimizationAggressivePass>(fuseFQAndMulWithNonConstInput, log);
}
