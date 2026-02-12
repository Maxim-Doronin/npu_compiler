//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/singleton_initializer.hpp"

namespace vpux::VPU::arch50xx {

/** @brief Creates and sets 50XX factories and utilities in the singleton cache for the given MLIR context. */
void initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform> platform);

}  // namespace vpux::VPU::arch50xx
