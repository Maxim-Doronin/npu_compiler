//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_factory.hpp"

namespace vpux::VPU {

/**
 * @brief Static class for encapsulating cost model factory-related objects.
 *
 * This class manages the singleton instance of the cost model factory,
 * ensuring thread safety and proper initialization.
 */
class CostModelConfig {
private:
    static std::map<ArchKind, std::unique_ptr<ICostModelFactory>>& _getFactories();

    static std::mutex& _getCostModelFactoryMutex() {
        static std::mutex mtx;
        return mtx;
    }

    /**
     * @brief Get the factory for the specified architecture
     *
     * @param arch Architecture kind
     * @return const ICostModelFactory&
     */
    static const ICostModelFactory& getFactory(ArchKind arch);

public:
    /**
     * @brief Set the factory for the specified architecture
     *
     * @param arch Architecture kind
     */
    static void setFactory(ArchKind arch);

    /**
     * @brief Create a cost model for the specified architecture
     *
     * @param arch Architecture kind
     * @return std::shared_ptr<VPUNN::VPUCostModel>
     */
    static std::shared_ptr<VPUNN::VPUCostModel> createCostModel(ArchKind arch) {
        return getFactory(arch).createCostModel();
    }

    /**
     * @brief Create a layer cost model for the specified architecture
     *
     * @param arch Architecture kind
     * @return std::shared_ptr<VPUNN::VPULayerCostModel>
     */
    static std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel(ArchKind arch) {
        return getFactory(arch).createLayerCostModel();
    }
};

}  // namespace vpux::VPU
