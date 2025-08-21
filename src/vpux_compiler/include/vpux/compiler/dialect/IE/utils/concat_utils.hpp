//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {
mlir::DenseSet<int64_t> getConcatAxes(IE::ConcatOp concatOp);
std::optional<std::pair<Dim, Shape>> inferOutputShapeAfterAffineReshapeBeforeConcat(mlir::Value curInput,
                                                                                    IE::ConcatOp concatOp,
                                                                                    IE::AffineReshapeOp reshapeOp);
mlir::ArrayAttr inferConcatOffsets(ArrayRef<ShapeRef> concatInShapes, const Dim concatDim, mlir::MLIRContext* ctx);

// TODO: E#159557 refactor initiative
mlir::Value createPaddingConstForConcat(ArrayRef<int64_t> constShape, mlir::Location loc,
                                        vpux::NDTypeInterface inputType, double padValue,
                                        mlir::PatternRewriter& rewriter);
const mlir::ArrayAttr inferOffsetsAttrWithAxis(IE::ConcatOp origOp, int64_t& axis);
std::optional<vpux::Dim> getConcatAxis(IE::ConcatOp concatOp);
}  // namespace IE
}  // namespace vpux
