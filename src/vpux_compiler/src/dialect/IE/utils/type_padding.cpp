//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Support/LogicalResult.h>

mlir::LogicalResult vpux::IE::unpadInputShape(SmallVector<int64_t>& shape, mlir::ArrayAttr inputPaddingAttr,
                                              mlir::Location loc) {
    if (inputPaddingAttr == nullptr) {
        return mlir::success();
    }
    const auto inputPadding = parseIntArrayAttr<int64_t>(inputPaddingAttr);
    if (inputPadding.size() != shape.size()) {
        return errorAt(loc, "Input padding '{0}' should have the same number of dimensions as the input shape '{1}'",
                       inputPadding, shape);
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] -= inputPadding[i];
    }
    return mlir::success();
}

mlir::LogicalResult vpux::IE::padOutputShape(SmallVector<int64_t>& shape, mlir::ArrayAttr outputPaddingAttr,
                                             mlir::Location loc) {
    if (outputPaddingAttr == nullptr) {
        return mlir::success();
    }
    const auto outputPadding = parseIntArrayAttr<int64_t>(outputPaddingAttr);
    if (outputPadding.size() != shape.size()) {
        return errorAt(loc, "Output padding '{0}' should have the same number of dimensions as the output shape '{1}'",
                       outputPadding, shape);
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] += outputPadding[i];
    }
    return mlir::success();
}

mlir::LogicalResult vpux::IE::checkPadding(mlir::ArrayAttr paddingAttr, mlir::Type type) {
    if (paddingAttr == nullptr) {
        return mlir::success();
    }
    const auto padding = parseIntArrayAttr<int64_t>(paddingAttr);
    const auto rank = static_cast<size_t>(mlir::cast<NDTypeInterface>(type).getRank());
    if (padding.size() != rank) {
        return mlir::failure();
    }
    return mlir::success();
}
