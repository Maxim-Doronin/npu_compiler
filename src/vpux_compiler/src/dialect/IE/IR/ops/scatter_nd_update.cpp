//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ScatterNDUpdateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ScatterNDUpdateOpAdaptor scatter(operands, attrs, prop);
    if (mlir::failed(scatter.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(scatter.getInput().getType());
    const auto tensorAttr = vpux::getTensorAttr(ctx, vpux::getOrder(inType), /*memSpace=*/nullptr, getBounds(inType));
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), tensorAttr);

    return mlir::success();
}
