//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/attributes/content.hpp"

namespace vpux::Const {

std::optional<mlir::Type> inferElemTypeAffineReshape(ShapeRef inputShape, mlir::Type inputElementType,
                                                     const SmallVector<SmallVector<int64_t>>& dimMapping,
                                                     ArrayRef<int64_t> shapeValue);

std::optional<DimsOrder> inferAffineReshapeOutputLayout(const DimArr& inPerm, mlir::ArrayAttr dimMapAttr);

/// For a [..., #const.AffineReshape, #const.SubView, ...] transformation pair this function
/// returns the new offset and shape of the SubView if that SubView can be ordered before AffineReshape or failure if
/// this is not possible.
mlir::FailureOr<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>>> swapAffineReshapeAndSubView(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape,
        const SmallVector<SmallVector<int64_t>>& dimMapping, ArrayRef<int64_t> subViewOffset,
        ArrayRef<int64_t> subViewShape);

}  // namespace vpux::Const
