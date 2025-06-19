//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"

namespace vpux {
namespace VPU {

static constexpr unsigned int VPUNN_CACHE_SIZE = 8156;

/**
 * @brief Interface for creating VPUNN Cost Models
 */
class ICostModelFactory {
public:
    virtual ~ICostModelFactory() = default;

    /**
     * @brief Create a VPUNN Cost Model
     *
     * @return std::shared_ptr<VPUNN::VPUCostModel>
     */
    virtual std::shared_ptr<VPUNN::VPUCostModel> createCostModel() const = 0;

    /**
     * @brief Create a VPUNN Layer Cost Model
     *
     * @return std::shared_ptr<VPUNN::VPULayerCostModel>
     */
    virtual std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel() const = 0;
};

}  // namespace VPU
}  // namespace vpux
