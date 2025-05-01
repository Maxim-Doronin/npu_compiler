//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GatherNDOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GatherNDOpAdaptor gatherND(operands, attrs, prop);
    if (mlir::failed(gatherND.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(gatherND.getInput().getType());
    auto originalShapeOptional = gatherND.getOriginalShape();
    vpux::Shape inputShape = originalShapeOptional.has_value()
                                     ? vpux::Shape(parseIntArrayAttr<int64_t>(originalShapeOptional.value()))
                                     : vpux::Shape(inType.getShape());
    const auto indicesShape = mlir::cast<mlir::ShapedType>(gatherND.getIndices().getType()).getShape();
    const auto batchDims = gatherND.getBatchDims();
    const auto lastIndices = indicesShape.back();
    const auto inputRank = static_cast<int64_t>(inputShape.size());

    SmallVector<int64_t> outShape;
    outShape.append(indicesShape.begin(), indicesShape.end() - 1);
    if (batchDims + lastIndices != inputRank) {
        outShape.append(inputShape.begin() + batchDims + lastIndices, inputShape.end());
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::IE::GatherNDOp::verify() {
    const auto op = getOperation();
    const auto inType = mlir::cast<mlir::ShapedType>(getInput().getType());
    auto originalShapeOptional = getOriginalShape();

    vpux::Shape inputShape = originalShapeOptional.has_value()
                                     ? vpux::Shape(parseIntArrayAttr<int64_t>(originalShapeOptional.value()))
                                     : vpux::Shape(inType.getShape());

    const auto indicesShape = mlir::cast<mlir::ShapedType>(getIndices().getType()).getShape();
    const auto batchDims = getBatchDims();
    const auto lastIndices = indicesShape.back();
    const auto inputRank = static_cast<int64_t>(inputShape.size());
    const auto indicesRank = static_cast<int64_t>(indicesShape.size());

    if (batchDims >= inputRank) {
        return errorAt(op, "batch_dims {0} exceeds input rank {1}", batchDims, inputRank);
    }

    if (batchDims >= indicesRank) {
        return errorAt(op, "batch_dims {0} exceeds indices rank {1}", batchDims, inputRank);
    }

    if (batchDims + lastIndices > inputRank) {
        return errorAt(op, "Slice index is out of bound");
    }

    for (size_t i = 0; i < static_cast<size_t>(batchDims); i++) {
        if (inputShape[Dim(i)] != indicesShape[i]) {
            return errorAt(op, "Batch dimensions of data and indices must be the same");
        }
    }

    return mlir::success();
}

//
// build
//

void vpux::IE::GatherNDOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                 ::mlir::Value indices, ::mlir::IntegerAttr batch_dims) {
    build(builder, state, input, indices, batch_dims, /*original_shape=*/{});
}
