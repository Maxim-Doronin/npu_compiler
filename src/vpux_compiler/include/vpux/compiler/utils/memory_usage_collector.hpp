//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {

void addMemoryUsageCollector(mlir::PassManager& pm, Logger log);

};
