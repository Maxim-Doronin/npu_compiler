//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/mc_strategy_getter.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

namespace vpux::VPU {

/*
   Find right class to get strategies for particular platform
*/
std::unique_ptr<StrategyGetterBase> createMCStrategyGetter(config::ArchKind arch, int64_t numClusters);

}  // namespace vpux::VPU
