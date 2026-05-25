//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DequantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DequantizeOpAdaptor dequantize(operands, attrs, prop);
    if (mlir::failed(dequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(dequantize.getInput().getType());
    const auto dstElemType = dequantize.getDstElemType();
    const auto outDesc = vpux::getTensorAttr(inType);

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);
    return mlir::success();
}

//
// ShaveCodeGenSupportedOpInterface
//

bool vpux::IE::DequantizeOp::shouldJITCompile() {
    auto inType = getInput().getType().getElementType();

    return mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(inType);
}
