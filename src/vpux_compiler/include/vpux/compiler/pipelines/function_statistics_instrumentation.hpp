//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

namespace mlir {
class PassManager;
}  // namespace mlir

namespace vpux {

void addFunctionStatisticsInstrumentation(mlir::PassManager& pm, const Logger& log);

}  // namespace vpux
