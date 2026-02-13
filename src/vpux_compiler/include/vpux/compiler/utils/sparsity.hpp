//
// Copyright (C) 2023-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"

namespace vpux::Const {
class Content;
}  // namespace vpux::Const

namespace mlir {
class Type;
}  // namespace mlir

namespace vpux {

// Get sparsify values and update element type to storage type
std::vector<int64_t> getSparsifyValues(mlir::Type& inputElementType);
int64_t getValuesPerSparsityBit(mlir::Type& elementType);
SmallVector<int64_t> countNonSparseElementsPerOC(const Const::Content& content, mlir::Type elementType);

}  // namespace vpux
