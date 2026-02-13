//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LogSoftmaxTopKOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LogSoftmaxTopKOpAdaptor logSoftmaxTopK(operands, attrs, prop);
    if (mlir::failed(logSoftmaxTopK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(logSoftmaxTopK.getInput().getType());
    const auto dstElemType = logSoftmaxTopK.getDstElemType();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);

    auto inputShape = inType.getShape();
    SmallVector<int64_t> topKShape(inputShape.begin(), inputShape.end());
    topKShape.back() = 1;  // Set width dimension to 1

    auto si64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
    inferredReturnShapes.emplace_back(topKShape, si64Type);

    return mlir::success();
}
