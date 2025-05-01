//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {

// Get sparsify value and update element type to storage type
int64_t getSparsifyValue(mlir::Type& inputElementType);
int64_t getValuesPerSparsityBit(mlir::Type& elementType);
SmallVector<int64_t> countNonSparseElementsPerOC(const Const::Content& content, mlir::Type elementType);

/*
 * Check if any of the activation input and output is sparse tensor type
 */
bool isActSparseOp(mlir::Operation* op);

}  // namespace vpux
