//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"  // ensure config::getArch(Platform) overload prevails

#include <mlir/IR/MLIRContext.h>

#include <variant>

namespace vpux::config {

using PlatformOrArch = std::variant<Platform, ArchKind>;

inline ArchKind getArch(PlatformOrArch target) {
    if (std::holds_alternative<Platform>(target)) {
        return getArch(std::get<Platform>(target));
    } else {
        return std::get<ArchKind>(target);
    }
}

//
// IConstraintsInitializer
//

class IConstraintsInitializer {
public:
    virtual void initialize(mlir::MLIRContext* context, PlatformOrArch target) = 0;
    virtual ~IConstraintsInitializer() = default;
};

}  // namespace vpux::config
