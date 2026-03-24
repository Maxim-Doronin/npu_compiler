//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::NonZeroOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NonZeroOpAdaptor nonZero(operands, attrs, prop);
    if (mlir::failed(nonZero.verify(loc))) {
        return mlir::failure();
    }

    auto inType = mlir::cast<vpux::NDTypeInterface>(nonZero.getInput().getType());
    const auto inRank = inType.getRank();
    const auto outShape = Shape{inRank, mlir::ShapedType::kDynamic};

    const auto numElements = inType.getNumElements();

    const auto typeComponents = TypeComponents()
                                        .setShape(outShape)
                                        .setDimsOrder(DimsOrder::fromNumDims(outShape.size()))
                                        .setElementType(nonZero.getDstElemType())
                                        .setBounds(Bounds{inRank, numElements});

    auto outType = inType.changeTypeComponents(typeComponents);

    const auto encoding = mlir::cast<mlir::RankedTensorType>(outType).getEncoding();

    inferredReturnShapes.emplace_back(outShape.raw(), nonZero.getDstElemType(), encoding);

    return mlir::success();
}
