//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/singleton_initializer.hpp"

namespace vpux::VPU::arch40xx {

/** @brief Creates and sets 40XX factories and utilities in the singleton cache for the given MLIR context. */
void initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform> platform);

}  // namespace vpux::VPU::arch40xx
