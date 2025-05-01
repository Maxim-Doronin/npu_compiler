//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/unsqueeze.hpp"

#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> propagateShape(mlir::Location loc, ArrayRef<int64_t> inShape,
                                                     ArrayRef<int64_t> axes) {
    SmallVector<int64_t> outShape(inShape.size() + axes.size());

    size_t inInd = 0;
    size_t axesInd = 0;
    for (auto outInd : irange(outShape.size())) {
        if (axesInd < axes.size()) {
            const auto nextAxisInd = checked_cast<size_t>(axes[axesInd]);

            if (nextAxisInd < outInd) {
                return errorAt(loc, "Axis '{0}' occurred twice", nextAxisInd);
            }

            if (nextAxisInd == outInd) {
                outShape[outInd] = 1;
                ++axesInd;
                continue;
            }
        }

        if (inInd < inShape.size()) {
            outShape[outInd] = inShape[inInd];
            ++inInd;
            continue;
        }
    }
    if (inInd != inShape.size() || axesInd != axes.size()) {
        return errorAt(loc, "Inconsistent parameters");
    }

    return outShape;
}

mlir::FailureOr<SmallVector<int64_t>> propagateDynamicAttr(mlir::Location loc, mlir::Value value,
                                                           ArrayRef<int64_t> axes) {
    auto type = value.getType();
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        const auto bounds = boundedType.getBounds();
        const auto outBounds = vpux::IE::propagateShape(loc, bounds.raw(), axes);
        if (mlir::failed(outBounds)) {
            return mlir::failure();
        }
        return outBounds.value();
    }

    if (auto dynamicDimsMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type)) {
        const auto dynamicDimsMask = dynamicDimsMaskType.getDynamicDimsMask();

        const auto invertDimsMask = [](int64_t dim) {
            return int64_t{dim == 0};
        };

        // propagateShape will put '1' into unsqueezed dimensions. Those dimensions will be static.
        // To re-use the function for dynamicDimsMask, we first invert the mask to swap dim representation
        // (1 - static dim, 0 - dynamic dim) and then after the unsqueeze invert them back
        const auto invertedMask = to_small_vector(dynamicDimsMask | transformed(invertDimsMask));
        const auto outInvertedDimsMask = vpux::IE::propagateShape(loc, invertedMask, axes);
        if (mlir::failed(outInvertedDimsMask)) {
            return mlir::failure();
        }

        auto outDimsMask = to_small_vector(outInvertedDimsMask.value() | transformed(invertDimsMask));
        return outDimsMask;
    }

    return SmallVector<int64_t>{};
}

}  // namespace IE
}  // namespace vpux
