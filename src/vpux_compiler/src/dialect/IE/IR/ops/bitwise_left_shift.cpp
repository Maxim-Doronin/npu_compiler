//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::BitwiseLeftShiftOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BitwiseLeftShiftOpAdaptor bitwiseLeftShift(operands, attrs, prop);
    if (mlir::failed(bitwiseLeftShift.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::ShapedType>(bitwiseLeftShift.getInput1().getType());
    const auto in2Type = mlir::cast<mlir::ShapedType>(bitwiseLeftShift.getInput2().getType());

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), bitwiseLeftShift.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.value(), in1Type.getElementType());
    }

    return mlir::success();
}
