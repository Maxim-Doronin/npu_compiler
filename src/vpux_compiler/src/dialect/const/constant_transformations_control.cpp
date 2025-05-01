//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/constant_transformations_control.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"

namespace vpux::Const {
namespace {
SmallVector<LazyFoldingOptions::OptimizeConstTransformationsFunc> getDefaultOptimizations(mlir::Type baseType) {
    auto moveSubViewBefore = [=](SmallVector<Const::TransformAttrInterface>& transformations,
                                 details::optimization::TransformAttrPos& currPos) {
        return details::moveSubViewBefore(transformations, currPos, baseType);
    };
    auto moveReshapeBefore = [=](SmallVector<Const::TransformAttrInterface>& transformations,
                                 details::optimization::TransformAttrPos& currPos) {
        return details::moveReshapeBefore(transformations, currPos, baseType);
    };
    auto fuseConsecutiveTransformations = [=](SmallVector<Const::TransformAttrInterface>& transformations,
                                              details::optimization::TransformAttrPos& currPos) {
        return details::fuseConsecutiveTransformations(transformations, currPos, baseType);
    };
    auto foldTransformation = [=](SmallVector<Const::TransformAttrInterface>& transformations,
                                  details::optimization::TransformAttrPos& currPos) {
        return details::foldTransformation(transformations, currPos, baseType);
    };

    return {fuseConsecutiveTransformations, foldTransformation, moveSubViewBefore, moveReshapeBefore,
            details::moveTransformationIntoFuse};
}
}  // namespace

LazyFoldingOptions::LazyFoldingOptions(): getFoldingSequenceOptimizations(getDefaultOptimizations) {
}
}  // namespace vpux::Const
