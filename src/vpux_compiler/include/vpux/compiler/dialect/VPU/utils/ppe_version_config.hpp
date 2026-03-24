//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"

#include <mlir/IR/DialectInterface.h>

namespace vpux::VPU {

/** @brief Singleton container for architecture-specific PPE factory. */
class PPEVersionConfig : public mlir::DialectInterface::Base<PPEVersionConfig> {
private:
    std::unique_ptr<IPpeFactory> _factory;

public:
    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PPEVersionConfig)

    PPEVersionConfig(mlir::Dialect* dialect): Base(dialect) {
    }

    void setPpeFactory(std::unique_ptr<IPpeFactory> factory) {
        _factory = std::move(factory);
    }

    const IPpeFactory& getFactory() const {
        VPUX_THROW_WHEN(_factory == nullptr, "PpeFactory not initialized");
        return *_factory;
    }

    template <typename DstT, std::enable_if_t<std::is_pointer_v<DstT>, bool> = true>
    auto getFactoryAs() const {
        using ConstDstPtrT = std::add_pointer_t<std::add_const_t<std::remove_pointer_t<DstT>>>;
        return dynamic_cast<const ConstDstPtrT>(_factory.get());
    }

    template <typename DstT, std::enable_if_t<!std::is_pointer_v<DstT>, bool> = true>
    const DstT& getFactoryAs() const {
        const auto* casted = dynamic_cast<const DstT*>(_factory.get());
        VPUX_THROW_WHEN(casted == nullptr, "Failed to cast the default PpeFactory instance to the required type");
        return *casted;
    }

    PPEAttr retrievePPEAttribute(mlir::Operation* operation) const {
        return _factory->retrievePPEAttribute(operation);
    }
};

void setPpeFactory(mlir::MLIRContext* context, std::unique_ptr<IPpeFactory> ppeFactory);
const IPpeFactory& getPpeFactory(mlir::MLIRContext* context);

PPEVersionConfig& getPpeConfig(mlir::MLIRContext* context);

}  // namespace vpux::VPU
