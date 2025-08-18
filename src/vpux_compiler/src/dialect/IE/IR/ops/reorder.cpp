//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReorderOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReorderOpAdaptor reorder(operands, attrs, prop);
    if (mlir::failed(reorder.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(reorder.getInput().getType());
    const auto tensorAttr = vpux::getTensorAttr(ctx, reorder.getDstOrder(), /*memSpace=*/nullptr, getBounds(inType));
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), tensorAttr);

    return mlir::success();
}

//
// Canonicalization
//

namespace {

#include <vpux/compiler/dialect/IE/reorder.hpp.inc>

}  // namespace

void vpux::IE::ReorderOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    populateWithGenerated(patterns);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReorderOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    if (const auto cst = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return static_cast<Const::ContentAttr>(cst).transform().reorder(DimsOrder::fromValue(getOutput())).get();
    }

    return nullptr;
}
