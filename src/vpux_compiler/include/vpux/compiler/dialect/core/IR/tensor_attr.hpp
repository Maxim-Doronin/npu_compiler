//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

namespace vpux {

//
// TensorAttr
//

class TensorAttr : public mlir::DictionaryAttr {
public:
    using mlir::DictionaryAttr::DictionaryAttr;

public:
    static bool classof(mlir::Attribute attr);

public:
    static TensorAttr get(mlir::MLIRContext* context, mlir::AffineMapAttr order, vpux::IndexedSymbolAttr memSpace,
                          BoundsRef bounds, DynamicDimsMaskRef dynamicDimsMask);

public:
    mlir::AffineMapAttr getOrder() const;
    vpux::IndexedSymbolAttr getMemSpace() const;
    Bounds getBounds() const;
    DynamicDimsMask getDynamicDimsMask() const;
};

//
// Helpers
//

TensorAttr getTensorAttr(mlir::MLIRContext* ctx, mlir::AffineMapAttr order, vpux::IndexedSymbolAttr memSpace,
                         BoundsRef bounds = {}, DynamicDimsMaskRef dynamicDimsMask = {});
TensorAttr getTensorAttr(mlir::MLIRContext* ctx, mlir::AffineMap order, vpux::IndexedSymbolAttr memSpace,
                         BoundsRef bounds = {}, DynamicDimsMaskRef dynamicDimsMask = {});
TensorAttr getTensorAttr(mlir::MLIRContext* ctx, vpux::DimsOrder order, vpux::IndexedSymbolAttr memSpace,
                         BoundsRef bounds = {}, DynamicDimsMaskRef dynamicDimsMask = {});

// A helper function to reduce code duplication in dynamic-shape-related cases
TensorAttr getTensorAttr(mlir::RankedTensorType type, mlir::AffineMap order, IndexedSymbolAttr memSpace,
                         mlir::ArrayRef<int64_t> dynamicAttr);

// Creates a TensorAttr corresponding to the given type which stores the provided order, memSpace and a possibly dynamic
// shape. If the given shape type (BoundedShape/DimsMaskedShape) doesn't match the expected representation
// (Bound/DynamicDimsMask), it will be automatically converted.
template <typename ShapeType>
TensorAttr getTensorAttr(mlir::RankedTensorType type, mlir::AffineMap order, IndexedSymbolAttr memSpace,
                         const ShapeType& shape) {
    if constexpr (std::is_same_v<ShapeType, Shape>) {
        VPUX_THROW_WHEN(mlir::isa<Core::BoundedTensorType>(type) || mlir::isa<Core::DynamicDimsMaskTensorType>(type),
                        "Cannot create tensor attribute from dynamic-shaped type: {0} without bounds or dims mask",
                        type);
        return vpux::getTensorAttr(type.getContext(), order, memSpace);

    } else {
        if (mlir::isa<Core::BoundedTensorType>(type)) {
            return vpux::getTensorAttr(type.getContext(), order, memSpace,
                                       shape.template toRepresentationOf<BoundedShape>());
        }
        if (mlir::isa<Core::DynamicDimsMaskTensorType>(type)) {
            return vpux::getTensorAttr(type.getContext(), order, memSpace, Bounds(),
                                       shape.template toRepresentationOf<DimsMaskedShape>());
        }
        VPUX_THROW("Cannot create a static tensor attribute from a dynamic shape, bounds would be discarded: {0}",
                   shape);
    }
}

TensorAttr getTensorAttr(mlir::RankedTensorType type);

mlir::AffineMap getOrder(mlir::RankedTensorType type);
vpux::IndexedSymbolAttr getMemorySpace(mlir::RankedTensorType type);
Bounds getBounds(mlir::Type type);
DynamicDimsMask getDynamicDimsMask(mlir::Type type);

}  // namespace vpux
