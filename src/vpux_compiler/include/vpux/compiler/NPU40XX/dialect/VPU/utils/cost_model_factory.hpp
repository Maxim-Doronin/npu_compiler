//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_factory.hpp"

namespace vpux {
namespace VPU {
namespace arch40xx {

/**
 * @brief NPU40XX implementation of the Cost Model Factory
 */
class CostModelFactory final : public ICostModelFactory {
public:
    /**
     * @brief Create a VPUNN Cost Model for NPU40XX architecture
     *
     * @return std::shared_ptr<VPUNN::VPUCostModel>
     */
    std::shared_ptr<VPUNN::VPUCostModel> createCostModel() const override;

    /**
     * @brief Create a VPUNN Layer Cost Model for NPU40XX architecture
     *
     * @return std::shared_ptr<VPUNN::VPULayerCostModel>
     */
    std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel() const override;
};

}  // namespace arch40xx
}  // namespace VPU
}  // namespace vpux
