//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/mem_permute_processing_pipeline_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

void IE::MemPermuteProcessingPipelineStrategy::registerRewriters(RewriterRegistry& registry, Logger& log) const {
    SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(10);
    IE::registerSwapMemPermuteAndExpandRewriters(registry, benefitLevels, 0, log);
    IE::registerPropagateMemPermuteBeforeOpRewriters(registry, benefitLevels, 1, log);
    IE::registerOptimizeConcatWithConvRewriters(registry, benefitLevels, 2, log);
    if (_enableAdjustConvShapePass) {
        // This pass further optimizes the IR after OptimizeConcatWithConv is applied. It is not part of
        // OptimizeConcatWithConv as this is a consequence of optimizing Concat to Convolution. This pass generalizes
        // optimizations of Convolution.
        IE::registerAdjustConvolutionShapeRewriters(registry, benefitLevels, 3, log);
    }
    IE::registerSwapOperationsRewriters(registry, benefitLevels, 5, _seOpsEnabled, log);
    IE::registerInsertIdentityPoolBeforeOpRewriters(registry, benefitLevels, 8, log);
    IE::registerOptimizeInnermostConcatRewriters(registry, benefitLevels, 9, log);
    IE::registerFuseMemPermuteRewriters(registry, benefitLevels, 9, log);
}

std::unique_ptr<IDynamicRewriterStrategy> IE::createMemPermuteProcessingPipelineStrategy(
        mlir::func::FuncOp, bool seOpsEnabled, bool enableAdjustConvShapePass) {
    return std::make_unique<IE::MemPermuteProcessingPipelineStrategy>(seOpsEnabled, enableAdjustConvShapePass);
}
