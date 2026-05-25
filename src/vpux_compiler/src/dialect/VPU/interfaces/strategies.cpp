//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

using namespace vpux;

void vpux::VPU::setVPUStrategyFactory(mlir::MLIRContext* context, std::unique_ptr<VPU::StrategyFactory> factory) {
    auto& registeredInterface = getCache<VPU::StrategyFactoryCache, VPU::VPUDialect>(context);
    registeredInterface.setStrategyFactory(std::move(factory));
}

const std::unique_ptr<VPU::StrategyFactory>& vpux::VPU::getVPUStrategyFactory(mlir::MLIRContext* context) {
    auto& registeredInterface = getCache<VPU::StrategyFactoryCache, VPU::VPUDialect>(context);
    return registeredInterface.getStrategyFactory();
}
