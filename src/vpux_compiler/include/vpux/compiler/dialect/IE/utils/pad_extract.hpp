//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::ArrayAttr padValue, Logger log);
mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::Location loc, const mlir::Value& padValue,
                                                  const std::optional<mlir::ArrayAttr>& padAttr,
                                                  vpux::ShapeRef inputShape);

// Adjust paddings attributes for tiled input
template <typename ConcreteOp>
void adjustPaddings(ConcreteOp* op, const TilingInfo& inputTiling) {
    const auto& inputTilePads = inputTiling.pads;
    VPUX_THROW_UNLESS(inputTilePads.has_value(), "Missing tile information for paddings");

    mlir::ArrayAttr newPadsBeginAttr, newPadsEndAttr;

    if (inputTilePads->is5D) {
        const std::array<int64_t, 3> padsBegin = {inputTilePads->front, inputTilePads->top, inputTilePads->left};
        const std::array<int64_t, 3> padsEnd = {inputTilePads->back, inputTilePads->bottom, inputTilePads->right};

        newPadsBeginAttr = getIntArrayAttr(op->getContext(), padsBegin);
        newPadsEndAttr = getIntArrayAttr(op->getContext(), padsEnd);
    } else {
        const std::array<int64_t, 2> padsBegin = {inputTilePads->top, inputTilePads->left};
        const std::array<int64_t, 2> padsEnd = {inputTilePads->bottom, inputTilePads->right};

        newPadsBeginAttr = getIntArrayAttr(op->getContext(), padsBegin);
        newPadsEndAttr = getIntArrayAttr(op->getContext(), padsEnd);
    }

    op->setPadsBeginAttr(newPadsBeginAttr);
    op->setPadsEndAttr(newPadsEndAttr);
}

}  // namespace IE
}  // namespace vpux
