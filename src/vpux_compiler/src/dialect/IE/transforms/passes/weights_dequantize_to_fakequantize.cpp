//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"

//
// createWeightsDequantizeToFakeQuantizePass
//

void vpux::IE::registerWeightsDequantizeToFakeQuantizeRewriters(RewriterRegistry& registry,
                                                                ArrayRef<mlir::PatternBenefit> benefitLevels,
                                                                size_t index, mlir::func::FuncOp func, Logger log) {
    auto& strategyFactory = IE::getIEStrategyFactory(func.getContext());
    auto strategy = strategyFactory->getWeightsDequantizeToFakeQuantizeStrategy(benefitLevels, index);
    strategy->registerRewriters(registry, log);
}
