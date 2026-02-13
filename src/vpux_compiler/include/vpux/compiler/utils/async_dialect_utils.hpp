//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace mlir {
class Value;
class Type;
}  // namespace mlir

namespace vpux {

// Get the type of the async value. If the value is not an async value, return its original type.
mlir::Type getAsyncValueType(mlir::Value value);

}  // namespace vpux
