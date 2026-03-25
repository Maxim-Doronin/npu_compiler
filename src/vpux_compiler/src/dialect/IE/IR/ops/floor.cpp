//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::FloorOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FloorOpAdaptor floor(operands, attrs, prop);
    if (mlir::failed(floor.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(floor.getInput().getType());
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::FloorOp::fold(FoldAdaptor /*adaptor*/) {
    const auto inputType = mlir::cast<NDTypeInterface>(getInput().getType());
    const auto outputType = mlir::cast<NDTypeInterface>(getOutput().getType());

    if (inputType != outputType) {
        return nullptr;
    }

    if (const auto elemType = mlir::dyn_cast<mlir::IntegerType>(inputType.getElementType())) {
        if (elemType.isSignedInteger(32) || elemType.isSignedInteger(64)) {
            return getInput();
        }
    }

    return nullptr;
}
