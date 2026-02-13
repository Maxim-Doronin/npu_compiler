//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"

using namespace vpux;

void vpux::IE::setIEStrategyFactory(mlir::MLIRContext* context, std::unique_ptr<IE::StrategyFactory> factory) {
    auto& registeredInterface = getCache<IE::StrategyFactoryCache, IE::IEDialect>(context);
    registeredInterface.setStrategyFactory(std::move(factory));
}

const std::unique_ptr<IE::StrategyFactory>& vpux::IE::getIEStrategyFactory(mlir::MLIRContext* context) {
    auto& registeredInterface = getCache<IE::StrategyFactoryCache, IE::IEDialect>(context);
    return registeredInterface.getStrategyFactory();
}
