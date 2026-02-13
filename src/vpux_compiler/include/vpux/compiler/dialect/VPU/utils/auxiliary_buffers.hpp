//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

// Create an empty auxiliary buffer, which will be represented by a VPU.Empty operation
// This is intended for SHAVE operations that do not need the auxiliary buffer to be pre-initialized with a
// given value, as the VPU.Empty operation will be lowered directly to a memory allocation
mlir::Value createEmptyAuxiliaryBuffer(mlir::OpBuilder& builder, mlir::Location opLoc, mlir::Type type);

// Create a constant auxiliary buffer, initialized with the value zero
// This is intended for SHAVE operations that need the auxiliary buffer to be zero-initialized by the compiler
mlir::Value createConstantAuxiliaryBuffer(mlir::OpBuilder& builder, mlir::Location opLoc, mlir::Type type);

mlir::LogicalResult compareTypes(mlir::Location loc, mlir::Type actual, mlir::Type expected);

}  // namespace VPU
}  // namespace vpux
