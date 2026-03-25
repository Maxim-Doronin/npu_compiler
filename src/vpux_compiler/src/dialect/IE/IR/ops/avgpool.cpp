//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/Support/LogicalResult.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::AvgPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AvgPoolOpAdaptor avgPool(operands, attrs, prop);
    if (mlir::failed(avgPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(avgPool.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(avgPool.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(avgPool.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(avgPool.getStrides());
    const auto roundingType = avgPool.getRoundingType();

    auto inputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getInput().getType());
    auto inShapeInfo = ShapeInfo::fromNDType(inputType);

    std::optional<SmallVector<int64_t>> inputPadding = std::nullopt;
    std::optional<SmallVector<int64_t>> outputPadding = std::nullopt;
    auto inputRes = IE::verifyPaddingAttr(avgPool.getInputPaddingAttr(), inShapeInfo, inputPadding);
    auto outputRes = IE::verifyPaddingAttr(avgPool.getOutputPaddingAttr(), inShapeInfo, outputPadding);
    if (inputRes.failed()) {
        return errorAt(loc, "Input padding '{0}' should have the same number of dimensions as the input shape '{1}'",
                       inputPadding, inShapeInfo.shape);
    }
    if (outputRes.failed()) {
        return errorAt(loc, "Output padding '{0}' should have the same number of dimensions as the input shape '{1}'",
                       outputPadding, inShapeInfo.shape);
    }

    const auto outShapeInfo = inferAvgPoolOutputShape(inShapeInfo, windowStrides, dataPaddingBelow, dataPaddingAbove,
                                                      windowShape, inputPadding, outputPadding, roundingType);

    const auto outDesc =
            vpux::getTensorAttr(ctx, inputType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(outShapeInfo.bounds));

    inferredReturnShapes.emplace_back(outShapeInfo.shape, inputType.getElementType(), outDesc);

    return mlir::success();
}
