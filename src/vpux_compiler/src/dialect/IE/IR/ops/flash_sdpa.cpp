//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::FlashSDPAOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FlashSDPAOpAdaptor flashSdpa(operands, attrs, prop);
    if (mlir::failed(flashSdpa.verify(loc))) {
        return mlir::failure();
    }

    const auto toSTC = [](mlir::Value value) -> mlir::ShapedTypeComponents {
        auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
        return mlir::ShapedTypeComponents(type.getShape(), type.getElementType(), type.getEncoding());
    };

    inferredReturnShapes.append({toSTC(flashSdpa.getInputRunningOutput()), toSTC(flashSdpa.getInputRunningMax()),
                                 toSTC(flashSdpa.getInputRunningSum())});

    return mlir::success();
}
