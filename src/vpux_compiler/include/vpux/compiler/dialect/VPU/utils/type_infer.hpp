//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

mlir::LogicalResult inferReduceReturnTypes(mlir::Location loc, mlir::Value input, bool keepDims,
                                           SmallVector<int64_t>& axes,
                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes,
                                           mlir::ArrayAttr inputPadding = nullptr,
                                           mlir::ArrayAttr outputPadding = nullptr);
void inferPermuteReturnTypes(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                             SmallVectorImpl<mlir::Type>& inferredReturnTypes);

vpux::TensorAttr createTensorAttrFromType(vpux::NDTypeInterface inType);

}  // namespace VPU
}  // namespace vpux
