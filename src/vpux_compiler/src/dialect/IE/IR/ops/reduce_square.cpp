//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceSquareOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceSquareOpAdaptor reduceSquare(operands, attrs, prop);
    if (mlir::failed(reduceSquare.verify(loc))) {
        return mlir::failure();
    }
    if (reduceSquare.getAxes() != nullptr && reduceSquare.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceSquare.getAxes() == nullptr && !reduceSquare.getAxesValue().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceSquare.getInput();
    const auto keepDims = reduceSquare.getKeepDims();

    auto axesValue = IE::extractAxes(loc, reduceSquare);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceSquareOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}
