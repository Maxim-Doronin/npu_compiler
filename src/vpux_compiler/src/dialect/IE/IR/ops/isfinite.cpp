//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::IsFiniteOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::IsFiniteOpAdaptor isFinite(operands, attrs, prop);
    if (mlir::failed(isFinite.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::ShapedType>(isFinite.getInput().getType());

    inferredReturnShapes.emplace_back(in1Type.getShape(), getBool8Type(ctx));

    return mlir::success();
}
