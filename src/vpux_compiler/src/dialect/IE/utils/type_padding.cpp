//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <cstddef>
#include <cstdint>

void vpux::IE::padShape(MutableArrayRef<int64_t> shape, ArrayRef<int64_t> padding) {
    VPUX_THROW_WHEN(shape.size() != padding.size(),
                    "Shape and padding should have the same number of dimensions, got {0} and {1}", shape.size(),
                    padding.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] += padding[i];
    }
}

void vpux::IE::unpadShape(MutableArrayRef<int64_t> shape, ArrayRef<int64_t> padding) {
    VPUX_THROW_WHEN(shape.size() != padding.size(),
                    "Shape and padding should have the same number of dimensions, got {0} and {1}", shape.size(),
                    padding.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] -= padding[i];
    }
}

mlir::LogicalResult vpux::IE::unpadInputShape(MutableArrayRef<int64_t> shape, mlir::ArrayAttr inputPaddingAttr,
                                              mlir::Location loc) {
    if (inputPaddingAttr == nullptr) {
        return mlir::success();
    }
    const auto inputPadding = parseIntArrayAttr<int64_t>(inputPaddingAttr);
    if (inputPadding.size() != shape.size()) {
        return errorAt(loc, "Input padding '{0}' should have the same number of dimensions as the input shape '{1}'",
                       inputPadding, shape);
    }
    unpadShape(shape, inputPadding);
    return mlir::success();
}

mlir::LogicalResult vpux::IE::padOutputShape(MutableArrayRef<int64_t> shape, mlir::ArrayAttr outputPaddingAttr,
                                             mlir::Location loc) {
    if (outputPaddingAttr == nullptr) {
        return mlir::success();
    }
    const auto outputPadding = parseIntArrayAttr<int64_t>(outputPaddingAttr);
    if (outputPadding.size() != shape.size()) {
        return errorAt(loc, "Output padding '{0}' should have the same number of dimensions as the output shape '{1}'",
                       outputPadding, shape);
    }
    padShape(shape, outputPadding);
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

mlir::LogicalResult vpux::IE::verifyPaddingAttr(mlir::ArrayAttr paddingAttr, const ShapeInfo& shapeInfo,
                                                std::optional<SmallVector<int64_t>>& paddingOut) {
    if (paddingAttr == nullptr) {
        return mlir::success();
    }

    const auto rawPad = parseIntArrayAttr<int64_t>(paddingAttr);
    paddingOut = rawPad;
    if (rawPad.size() != shapeInfo.shape.size()) {
        return mlir::failure();
    }
    if (shapeInfo.isDynamic() && rawPad.size() != shapeInfo.bounds.size()) {
        return mlir::failure();
    }
    return mlir::success();
}
