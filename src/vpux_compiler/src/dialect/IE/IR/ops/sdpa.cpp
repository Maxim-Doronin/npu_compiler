//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SDPAOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SDPAOpAdaptor sdpa(operands, attrs, prop);
    if (mlir::failed(sdpa.verify(loc))) {
        return mlir::failure();
    }
    const auto inType = mlir::cast<vpux::NDTypeInterface>(sdpa.getInputQ().getType());
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());
    return mlir::success();
}
