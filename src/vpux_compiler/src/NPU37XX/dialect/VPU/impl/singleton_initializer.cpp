//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/utils/cost_model_shave_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"

using namespace vpux::VPU;

void arch37xx::initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform>) {
    const bool isShave2ApiUsedInVPUNN = false;

    setCostModelFactory(context, std::make_unique<arch37xx::CostModelFactory>());
    setShaveCostModelUtils(context, std::make_unique<arch37xx::CostModelShaveUtil>(isShave2ApiUsedInVPUNN));
}
