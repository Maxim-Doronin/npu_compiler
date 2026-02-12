//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_factory.hpp"

#include <mlir/IR/DialectInterface.h>

namespace vpux {
namespace VPU {

/** @brief Singleton container for various architecture-specific factories and utilities. */
class SingletonCache final : public mlir::DialectInterface::Base<SingletonCache> {
    std::unique_ptr<ICostModelFactory> _costModelFactory;
    std::unique_ptr<IShaveCostModelUtils> _shaveCostModelUtils;

public:
    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SingletonCache)

    SingletonCache(mlir::Dialect* dialect): Base(dialect) {
    }

    const ICostModelFactory& getCostModelFactory() const {
        assert(_costModelFactory != nullptr && "Cost model factory is not set");
        return *_costModelFactory;
    }

    void setCostModelFactory(std::unique_ptr<ICostModelFactory> costModelFactory) {
        _costModelFactory = std::move(costModelFactory);
    }

    const IShaveCostModelUtils& getShaveCostModelUtils() const {
        assert(_shaveCostModelUtils != nullptr && "Shave cost model utils is not set");
        return *_shaveCostModelUtils;
    }

    void setShaveCostModelUtils(std::unique_ptr<IShaveCostModelUtils> shaveCostModelUtils) {
        _shaveCostModelUtils = std::move(shaveCostModelUtils);
    }
};

/** @brief Sets the cost model factory in the singleton cache for the given MLIR context. */
void setCostModelFactory(mlir::MLIRContext* context, std::unique_ptr<ICostModelFactory> costModelFactory);

/**  @brief Gets the cost model factory from the singleton cache for the given MLIR context. */
const ICostModelFactory& getCostModelFactory(mlir::MLIRContext* context);

/** @brief Sets the shave cost model utilities in the singleton cache for the given MLIR context. */
void setShaveCostModelUtils(mlir::MLIRContext* context, std::unique_ptr<IShaveCostModelUtils> shaveCostModelUtils);

/** @brief Gets the shave cost model utilities from the singleton cache for the given MLIR context. */
const IShaveCostModelUtils& getShaveCostModelUtils(mlir::MLIRContext* context);

}  // namespace VPU
}  // namespace vpux
