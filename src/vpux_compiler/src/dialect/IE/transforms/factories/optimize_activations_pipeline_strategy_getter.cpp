//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/optimize_activations_pipeline_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

void IE::OptimizeActivationsPipelineStrategy::registerRewriters(RewriterRegistry& registry, Logger& log) const {
    SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(5);
    registerSwapOperationsRewriters(registry, benefitLevels, 0, _enableSEOps, log);
    registerInsertIdentityPoolBeforeOpRewriters(registry, benefitLevels, 1, log);
    registerSwapMaxpoolWithActivation(registry, benefitLevels, 2, log);
    registerFuseActivationOpsRewriters(registry, _enableFuseClamp, log);
}

std::unique_ptr<IDynamicRewriterStrategy> IE::createOptimizeActivationsPipelineStrategy(mlir::func::FuncOp,
                                                                                        bool enableSEOps,
                                                                                        bool enableFuseClamp) {
    return std::make_unique<IE::OptimizeActivationsPipelineStrategy>(enableSEOps, enableFuseClamp);
}
