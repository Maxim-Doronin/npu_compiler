//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GRUSequenceLastPartOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GRUSequenceLastPartOpAdaptor gru(operands, attrs, prop);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto initialStateType = mlir::cast<mlir::ShapedType>(gru.getInitialHiddenState().getType());
    const auto outputStateShape = initialStateType.getShape();
    const auto seqLength = gru.getSeqLength();
    SmallVector<int64_t> middleStateShape = {outputStateShape[0], outputStateShape[1], seqLength, outputStateShape[2]};

    inferredReturnShapes.emplace_back(middleStateShape, initialStateType.getElementType());
    inferredReturnShapes.emplace_back(outputStateShape, initialStateType.getElementType());

    return mlir::success();
}
