//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/utils/attributes_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

namespace vpux::Const {

mlir::FailureOr<SmallVector<int64_t>> getConstArrValue(mlir::Value input) {
    auto op = input.getDefiningOp<Const::DeclareOp>();
    if (op == nullptr) {
        return mlir::failure();
    }

    const auto content = op.getContent();

    return to_small_vector(content.getValues<int64_t>());
}

mlir::FailureOr<int64_t> getConstOrAttrValue(mlir::Value input, mlir::IntegerAttr attr) {
    return (input != nullptr) ? Const::getSplatValue<int64_t>(input) : attr.getValue().getSExtValue();
}

mlir::FailureOr<SmallVector<int64_t>> getConstOrArrAttrValue(mlir::Value input, mlir::ArrayAttr attr) {
    return (input != nullptr) ? getConstArrValue(input) : parseIntArrayAttr<int64_t>(attr);
}

}  // namespace vpux::Const
