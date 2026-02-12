//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassOptions.h>

namespace mlir {
class ModuleOp;
class PassManager;
class Operation;
}  // namespace mlir

namespace vpux {

enum class LocationsVerificationMode { OFF, FAST, FULL, THOROUGH };

LocationsVerificationMode getLocationsVerificationMode(mlir::ModuleOp moduleOp);

LocationsVerificationMode getLocationsVerificationMode(
        const mlir::detail::PassOptions::Option<std::string>& locationsVerificationMode);

void setLocationsVerificationMode(mlir::ModuleOp moduleOp, LocationsVerificationMode mode);

std::string stringifyLocationsVerificationMode(LocationsVerificationMode mode);

LocationsVerificationMode symbolizeLocationsVerificationMode(StringRef strMode);

void addLocationsVerifier(mlir::PassManager& pm);

mlir::LogicalResult verifyLocationsUniquenessFull(mlir::Operation* op, StringRef passName);
mlir::LogicalResult verifyLocationsUniquenessFast(mlir::Operation* op, StringRef passName);

// verifyLocations is a wrapper for verifyLocationsUniquenessFull and verifyLocationsUniquenessFast
// depending on the current locations verification mode in Module
mlir::LogicalResult verifyLocations(mlir::Operation* op, StringRef passName);

};  // namespace vpux
