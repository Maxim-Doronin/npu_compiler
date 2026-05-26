//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mlir/IR/DialectRegistry.h>

namespace vpux {
namespace VPU {

//
// DeviceVersion
//

struct DeviceVersion {
    // This is optional because lit-tests do not contain platform information
    std::optional<config::Platform> platform;

    // This field is used in order to select correct singleton initializer
    config::ArchKind arch{config::ArchKind::UNKNOWN};
};

/** @brief Adds dialect extension in order to initialize singletons for the specified architecture. */
void initializeSingletons(mlir::DialectRegistry& registry, const DeviceVersion& deviceVersion);

}  // namespace VPU
}  // namespace vpux
