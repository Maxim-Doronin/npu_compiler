//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"

#include <vpu_cost_model.h>

using namespace vpux::VPU;

void arch40xx::initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform>) {
    const bool isShave2ApiUsedInVPUNN = false;

    auto costModelFactory = std::make_unique<arch40xx::CostModelFactory>();
    auto costModel = costModelFactory->createCostModel();
    auto supportedOps = costModel->getShaveSupportedOperations(VPUNN::VPUDevice::VPU_4_0);

    setCostModelFactory(context, std::move(costModelFactory));
    setShaveCostModelUtils(context, std::make_unique<CostModelShaveUtil>(isShave2ApiUsedInVPUNN, supportedOps));
}
