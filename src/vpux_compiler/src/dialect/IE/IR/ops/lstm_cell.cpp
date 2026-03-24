//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LSTMCellOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LSTMCellOpAdaptor lstm(operands, attrs, prop);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(lstm.getInitialHiddenState().getType());

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputHiddenState
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputCellState

    return mlir::success();
}
