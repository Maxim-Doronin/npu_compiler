//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace IE {

bool hasDynamicShapeAttr(mlir::Value value);
bool hasDynamicTensors(mlir::Operation* op);
bool needsStaticShape(mlir::Operation* op);
bool isDynamicDataContiguous(vpux::ShapeRef shape, vpux::DimsOrder order);

template <typename T>
SmallVector<T> replaceDynamicDimsWithValue(const SmallVector<int64_t>& original, T value) {
    const auto originalRank = static_cast<int64_t>(original.size());
    const auto transformDim = [value](auto dim) -> T {
        return dim != mlir::ShapedType::kDynamic ? static_cast<T>(dim) : value;
    };

    SmallVector<T> transformed(originalRank);
    transform(original, std::begin(transformed), transformDim);

    return transformed;
}

}  // namespace IE

// Helpers for extracting the static shape from any shape type.
// May be used in templated contexts where the shape type is unknown.
Shape extractShape(const Shape& shape);
Shape extractShape(const BoundedShape& shape);
Shape extractShape(const DimsMaskedShape& shape);

// Helpers for splitting any shape type into its corresponding static Shape, Bounds and DynamicDimsMask representations.
// Dynamic representations other than the one used by the given shape type are left empty. May be used in templated
// contexts where the shape type is unknown.
std::tuple<Shape, Bounds, DynamicDimsMask> splitShapeAndRepresentation(const Shape& shape);
std::tuple<Shape, Bounds, DynamicDimsMask> splitShapeAndRepresentation(const BoundedShape& shape);
std::tuple<Shape, Bounds, DynamicDimsMask> splitShapeAndRepresentation(const DimsMaskedShape& shape);

// Invokes and returns the result of a dynamic-shape-friendly functor on the concrete shape
// of a tensor type, eliminating the need for call site type-switching.
template <typename Func, typename... Args>
decltype(auto) callOnShapeOf(NDTypeInterface type, Func func, Args&&... args) {
    if (const auto dynShapeType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        return func(dynShapeType.getDynamicShape(), std::forward<Args>(args)...);
    } else if (const auto dynShapeType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type)) {
        return func(dynShapeType.getDynamicShape(), std::forward<Args>(args)...);
    } else {
        // Note: For static-shaped types func(...) is given a ShapeRef (instead of Shape). This is not possible for the
        // other types since the shape and bounds/dims-mask arrays are stored separately.
        return func(type.getShape(), std::forward<Args>(args)...);
    }
}

}  // namespace vpux
