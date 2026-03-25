//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/split_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux {
namespace IE {

// Check the split dim size after splitOp is 1 to make it feasible to convert into TransposeOp
mlir::FailureOr<vpux::Dim> getSplitDimToShape1(IE::SplitOp splitOp) {
    const auto splitInputShape = getShape(splitOp.getInput());
    const auto splitDim = Dim(splitOp.getAxisValue().value());
    const auto splitNum = splitOp.getNumSplits();

    if (splitInputShape[splitDim] != splitNum) {
        return mlir::failure();
    }

    return splitDim;
}

}  // namespace IE
}  // namespace vpux
