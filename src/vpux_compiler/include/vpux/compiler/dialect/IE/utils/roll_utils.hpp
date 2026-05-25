//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {
namespace IE {

struct RollShiftAndAxes final {
    SmallVector<int64_t> shift;
    SmallVector<int64_t> axes;

    explicit RollShiftAndAxes(ArrayRef<int64_t> shiftRef, ArrayRef<int64_t> axesRef): shift(shiftRef), axes(axesRef) {
    }
};

mlir::FailureOr<RollShiftAndAxes> getShiftAndAxesForRollOp(mlir::Location loc, mlir::Value shiftValue,
                                                           mlir::Value axesValue, ShapeRef inputShape,
                                                           bool adjustSpatial = true);

}  // namespace IE
}  // namespace vpux
