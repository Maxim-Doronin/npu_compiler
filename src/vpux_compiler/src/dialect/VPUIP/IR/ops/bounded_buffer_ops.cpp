//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

using namespace vpux;

mlir::LogicalResult VPUIP::GroupBoundedBufferOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                                  std::optional<mlir::Location> optLoc,
                                                                  mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                                  mlir::OpaqueProperties props,
                                                                  mlir::RegionRange /*ranges*/,
                                                                  SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::GroupBoundedBufferOpAdaptor op(operands, attrs, props);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto dataTy = op.getData().getType();
    const auto shapeTy = op.getDynamicShape().getType();

    inferredReturnTypes.push_back(VPUIP::BoundedBufferType::get(dataTy, shapeTy));

    return mlir::success();
}

mlir::ValueRange VPUIP::GroupBoundedBufferOp::getViewSources() {
    return getOperands();
}

mlir::LogicalResult VPUIP::UngroupBoundedBufferOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties props, mlir::RegionRange /*ranges*/,
        SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::UngroupBoundedBufferOpAdaptor op(operands, attrs, props);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto boundedBufferTy = mlir::cast<vpux::VPUIP::BoundedBufferType>(op.getInput().getType());
    inferredReturnTypes.push_back(boundedBufferTy.getData());
    inferredReturnTypes.push_back(boundedBufferTy.getDynamicShape());

    return mlir::success();
}

mlir::Value VPUIP::UngroupBoundedBufferOp::getViewSource(ptrdiff_t idx) {
    VPUX_THROW_UNLESS(idx == 0 || idx == 1,
                      "UngroupBoundedBufferOp should have one view source with two aliases, got {0} offset", idx);
    return getOperand();
}
