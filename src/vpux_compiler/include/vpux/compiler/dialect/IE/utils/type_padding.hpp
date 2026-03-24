//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <optional>

#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {
namespace IE {

void padShape(MutableArrayRef<int64_t> shape, ArrayRef<int64_t> padding);
void unpadShape(MutableArrayRef<int64_t> shape, ArrayRef<int64_t> padding);

mlir::LogicalResult unpadInputShape(MutableArrayRef<int64_t> shape, mlir::ArrayAttr inputPaddingAttr,
                                    mlir::Location loc);
mlir::LogicalResult padOutputShape(MutableArrayRef<int64_t> shape, mlir::ArrayAttr outputPaddingAttr,
                                   mlir::Location loc);

mlir::LogicalResult verifyPaddingAttr(mlir::ArrayAttr paddingAttr, const ShapeInfo& shapeInfo,
                                      std::optional<SmallVector<int64_t>>& paddingOut);

mlir::LogicalResult checkPadding(mlir::ArrayAttr paddingAttr, mlir::Type type);

}  // namespace IE
}  // namespace vpux
