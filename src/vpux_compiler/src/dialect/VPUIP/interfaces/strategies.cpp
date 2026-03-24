//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/strategies.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

using namespace vpux;

void vpux::VPUIP::setVPUIPStrategyFactory(mlir::MLIRContext* context, std::unique_ptr<VPUIP::StrategyFactory> factory) {
    auto& registeredInterface = getCache<VPUIP::StrategyFactoryCache, VPUIP::VPUIPDialect>(context);
    registeredInterface.setStrategyFactory(std::move(factory));
}

const std::unique_ptr<VPUIP::StrategyFactory>& vpux::VPUIP::getVPUIPStrategyFactory(mlir::MLIRContext* context) {
    auto& registeredInterface = getCache<VPUIP::StrategyFactoryCache, VPUIP::VPUIPDialect>(context);
    return registeredInterface.getStrategyFactory();
}
