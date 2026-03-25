//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/mc_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/mc_strategy_getter.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/mc_strategy_getter.hpp"

using namespace vpux::VPU;

std::unique_ptr<StrategyGetterBase> vpux::VPU::createMCStrategyGetter(config::ArchKind arch, int64_t numClusters) {
    if (numClusters == 1) {
        return std::make_unique<StrategyGetterBase>();
    }
    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return std::make_unique<arch37xx::StrategyGetter>();
    }
    default: {
        return std::make_unique<arch40xx::StrategyGetter>(numClusters);
    }
    }
}
