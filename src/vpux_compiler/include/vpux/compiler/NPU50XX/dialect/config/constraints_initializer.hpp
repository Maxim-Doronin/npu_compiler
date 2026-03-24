//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/constraints_initializer.hpp"

namespace vpux::config {

//
// Initializing architecture-specific constraints in context
//

class ConstraintsInitializer50XX final : public IConstraintsInitializer {
public:
    void initialize(mlir::MLIRContext* context, PlatformOrArch target) override;
};

}  // namespace vpux::config
