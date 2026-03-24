//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GridSampleOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GridSampleOpAdaptor gridSample(operands, attrs, prop);

    if (mlir::failed(gridSample.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(gridSample.getInput().getType());
    const auto inShape = inType.getShape();

    const auto gridType = mlir::cast<mlir::ShapedType>(gridSample.getGrid().getType());
    const auto gridShape = gridType.getShape();

    SmallVector<int64_t> outShape = {inShape[0], inShape[1], gridShape[1], gridShape[2]};

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
