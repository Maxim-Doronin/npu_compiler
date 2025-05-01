//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/cast_utils.hpp"

mlir::LogicalResult vpux::IERT::QuantizeCastOp::verify() {
    auto inputStrides = getStrides(getInput());
    auto outputStrides = getStrides(getOutput());
    if (inputStrides != outputStrides) {
        return errorAt(getLoc(), "QuantizeCastOp input and output must have the same strides, but got {0} and {1}",
                       inputStrides, outputStrides);
    }

    auto inputShape = getShape(getInput());
    auto outputShape = getShape(getOutput());
    if (inputShape != outputShape) {
        return errorAt(getLoc(), "QuantizeCastOp input and output must have the same shape, but got {0} and {1}",
                       inputShape, outputShape);
    }

    const auto dstElemType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType()).getElementType();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getElementType();

    return vpux::isQuantizeCastValid(getLoc(), inputType, dstElemType);
}

mlir::Value vpux::IERT::QuantizeCastOp::getViewSource() {
    return getInput();
}

mlir::OpFoldResult vpux::IERT::QuantizeCastOp::fold(FoldAdaptor) {
    return getInput().getType() == getOutput().getType() ? getInput() : mlir::TypedValue<mlir::MemRefType>{nullptr};
}
