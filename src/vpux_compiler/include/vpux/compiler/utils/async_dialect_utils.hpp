//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace mlir {
class Location;
class OpBuilder;
class Value;
class Type;
}  // namespace mlir

namespace vpux {

// Get the type of the async value. If the value is not an async value, return its original type.
mlir::Type getAsyncValueType(mlir::Value value);

mlir::Value allocateSpillReadBuffer(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value bufferToSpill);

}  // namespace vpux
