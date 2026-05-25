//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"

//
// registerWeightsDequantizeToDynamicDequantizeRewriters
//

void vpux::IE::registerWeightsDequantizeToDynamicDequantizeRewriters(RewriterRegistry& registry,
                                                                     ArrayRef<mlir::PatternBenefit> benefitLevels,
                                                                     size_t index, mlir::func::FuncOp func,
                                                                     Logger log) {
    const auto& strategyFactory = IE::getIEStrategyFactory(func.getContext());
    auto strategy = strategyFactory->getWeightsDequantizeToDynamicDequantizeStrategy(benefitLevels, index);
    strategy->registerRewriters(registry, log);
}
