//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/utils/cost_model_shave_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"

using namespace vpux::VPU;

void arch50xx::initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform> platform) {
    const bool isShave2ApiUsedInVPUNN = false;

    setCostModelFactory(context, std::make_unique<arch50xx::CostModelFactory>(platform));
    setShaveCostModelUtils(context, std::make_unique<arch50xx::CostModelShaveUtil>(isShave2ApiUsedInVPUNN));
}
