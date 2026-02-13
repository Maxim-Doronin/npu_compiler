//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/constraints_initializer.hpp"

#include <mlir/IR/DialectRegistry.h>

namespace vpux {
class IStrategiesInitializer {
public:
    virtual void initialize(mlir::MLIRContext* context) = 0;
    virtual ~IStrategiesInitializer();
};
}  // namespace vpux

namespace vpux::config {

//
// registerConstraints
//

void registerConstraints(mlir::DialectRegistry& registry, PlatformOrArch target);

}  // namespace vpux::config

namespace vpux::IE {

void registerStrategies(mlir::DialectRegistry& registry, config::ArchKind arch);

}
