//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GatherElementsOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GatherElementsOpAdaptor gatherElements(operands, attrs, prop);
    if (mlir::failed(gatherElements.verify(loc))) {
        return mlir::failure();
    }

    const auto inIndicesType = mlir::cast<mlir::ShapedType>(gatherElements.getIndices().getType());
    const auto inInputType = mlir::cast<mlir::ShapedType>(gatherElements.getInput().getType());

    inferredReturnShapes.emplace_back(inIndicesType.getShape(), inInputType.getElementType());
    return mlir::success();
}
