//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace vpux::locverif {

std::unique_ptr<mlir::Pass> createStartLocationVerifierPass(
        const Logger& log, const mlir::detail::PassOptions::Option<std::string>& locationsVerificationMode);
std::unique_ptr<mlir::Pass> createStopLocationVerifierPass(const Logger& log);

void registerPasses();

}  // namespace vpux::locverif
