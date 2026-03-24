//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"

#include <vpu_cost_model.h>

using namespace vpux::VPU;

void arch50xx::initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform> platform) {
    const bool isShave2ApiUsedInVPUNN = false;

    auto costModelFactory = std::make_unique<arch50xx::CostModelFactory>(platform);
    auto costModel = costModelFactory->createCostModel();
    auto supportedOps = costModel->getShaveSupportedOperations(VPUNN::VPUDevice::NPU_5_0);

    setCostModelFactory(context, std::move(costModelFactory));
    setShaveCostModelUtils(context, std::make_unique<CostModelShaveUtil>(isShave2ApiUsedInVPUNN, supportedOps));
}

void arch50xx::initializePPEVersionConfig(mlir::MLIRContext* context) {
    setPpeFactory(context, std::make_unique<VPU::arch50xx::PpeFactory>());
}
