//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <openvino/op/group_conv.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::GroupTransposedConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupTransposedConvolutionOpAdaptor groupTransposedConv(operands, attrs, prop);
    if (mlir::failed(groupTransposedConv.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(groupTransposedConv.getInput().getType());
    const auto inputShape = to_small_vector(inputType.getShape());
    const auto inputElemType = inputType.getElementType();
    const auto outputShape = groupTransposedConv.getOutputShape();
    const auto filterShape =
            to_small_vector(mlir::cast<vpux::NDTypeInterface>(groupTransposedConv.getFilter().getType()).getShape());

    if (outputShape != nullptr) {
        return errorAt(loc, "Explicit output shape is not implemented");
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(groupTransposedConv.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(groupTransposedConv.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(groupTransposedConv.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(groupTransposedConv.getDilations());
    const auto outputPadding = parseIntArrayAttr<int64_t>(groupTransposedConv.getSpatialOutputPadding());

    auto mlirOutputShape = inferTransposedGroupConvBackpropOutputShape(
            inputShape, filterShape, windowStrides, dataPaddingBelow, dataPaddingAbove, windowDilations, outputPadding);

    inferredReturnShapes.emplace_back(mlirOutputShape, inputElemType);

    return mlir::success();
}
