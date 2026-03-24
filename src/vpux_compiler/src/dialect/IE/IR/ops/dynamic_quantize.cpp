//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicQuantizeOpAdaptor quantize(operands, attrs, prop);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    auto scaleOrZpShape = SmallVector<int64_t>{1};
    if (quantize.getMin() != nullptr && quantize.getMax() != nullptr) {
        const auto minType = mlir::cast<mlir::ShapedType>(quantize.getMin().getType());
        scaleOrZpShape = SmallVector<int64_t>(minType.getRank(), 1);
    }

    const auto inType = mlir::cast<mlir::ShapedType>(quantize.getInput().getType());
    auto ui8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
    inferredReturnShapes.emplace_back(inType.getShape(), ui8Type);
    inferredReturnShapes.emplace_back(scaleOrZpShape, inType.getElementType());
    inferredReturnShapes.emplace_back(scaleOrZpShape, ui8Type);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::DynamicQuantizeOp::verify() {
    const auto isShapeSizeOne = [](mlir::Value tensor) {
        if (tensor == nullptr) {
            return true;
        }
        return getShape(tensor).totalSize() == 1;
    };

    if (!isShapeSizeOne(getMin())) {
        return errorAt(getLoc(), "The min input doesn't have a single element");
    }

    if (!isShapeSizeOne(getMax())) {
        return errorAt(getLoc(), "The max input doesn't have a single element");
    }

    if (!isShapeSizeOne(getScale())) {
        return errorAt(getLoc(), "The scale output doesn't have a single element");
    }

    if (!isShapeSizeOne(getZeroPoint())) {
        return errorAt(getLoc(), "The zero-point output doesn't have a single element");
    }

    return mlir::success();
}
