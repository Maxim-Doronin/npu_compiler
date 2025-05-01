//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Pass/PassManager.h>
#include "vpux/utils/logger/logger.hpp"

namespace vpux {

void addFunctionStatisticsInstrumentation(mlir::PassManager& pm, const Logger& log);

};
