//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Interfaces/InferTypeOpInterface.h>

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult unpadInputShape(SmallVector<int64_t>& shape, mlir::ArrayAttr inputPaddingAttr, mlir::Location loc);
mlir::LogicalResult padOutputShape(SmallVector<int64_t>& shape, mlir::ArrayAttr outputPaddingAttr, mlir::Location loc);

mlir::LogicalResult checkPadding(mlir::ArrayAttr paddingAttr, mlir::Type type);

}  // namespace IE
}  // namespace vpux
