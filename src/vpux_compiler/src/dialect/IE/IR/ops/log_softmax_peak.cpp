//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LogSoftmaxPeakOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LogSoftmaxPeakOpAdaptor logSoftmaxPeak(operands, attrs, prop);
    if (mlir::failed(logSoftmaxPeak.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(logSoftmaxPeak.getInput().getType());
    const auto dstElemType = logSoftmaxPeak.getDstElemType();
    auto si64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);

    auto inputShape = inType.getShape();
    SmallVector<int64_t> topKShape(inputShape.begin(), inputShape.end());
    topKShape.back() = 1;  // Set width dimension to 1

    inferredReturnShapes.emplace_back(topKShape, dstElemType);
    inferredReturnShapes.emplace_back(topKShape, si64Type);

    return mlir::success();
}
