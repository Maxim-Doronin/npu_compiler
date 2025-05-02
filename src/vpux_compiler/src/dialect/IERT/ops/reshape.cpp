//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IERT::GenericReshapeOp::verify() {
    const auto op = getOperation();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "Reshape input and output must have the same number of elements");
    }

    const auto inReqs = StrideReqs::compact(inType.getRank());
    const auto outReqs = StrideReqs::compact(outType.getRank());

    if (!inReqs.checkStrides(inType)) {
        return errorAt(op, "Input strides do not match requirements '{0}'", inType);
    }
    if (!outReqs.checkStrides(outType)) {
        return errorAt(op, "Output strides do not match requirements '{0}'", inType);
    }

    return mlir::success();
}

mlir::Value vpux::IERT::GenericReshapeOp::getViewSource() {
    return getInput();
}

mlir::OpFoldResult vpux::IERT::GenericReshapeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    if (getInput().getDefiningOp<GenericReshapeOp>() != nullptr) {
        return getOutput();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return static_cast<Const::ContentAttr>(cst).transform().reshape(getShape(getOutput())).get();
    }

    return nullptr;
}
