//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"

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
                          Bounds bounds, DynamicDimsMask dynamicDimsMask);

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
                         Bounds bounds = {}, DynamicDimsMask dynamicDimsMask = {});
TensorAttr getTensorAttr(mlir::MLIRContext* ctx, mlir::AffineMap order, vpux::IndexedSymbolAttr memSpace,
                         Bounds bounds = {}, DynamicDimsMask dynamicDimsMask = {});
TensorAttr getTensorAttr(mlir::MLIRContext* ctx, vpux::DimsOrder order, vpux::IndexedSymbolAttr memSpace,
                         Bounds bounds = {}, DynamicDimsMask dynamicDimsMask = {});

// A helper function to reduce code duplication in dynamic-shape-related cases
TensorAttr getTensorAttr(mlir::RankedTensorType type, mlir::AffineMap order, IndexedSymbolAttr memSpace,
                         mlir::ArrayRef<int64_t> dynamicAttr);

TensorAttr getTensorAttr(mlir::RankedTensorType type);

mlir::AffineMap getOrder(mlir::RankedTensorType type);
vpux::IndexedSymbolAttr getMemorySpace(mlir::RankedTensorType type);
Bounds getBounds(mlir::Type type);
DynamicDimsMask getDynamicDimsMask(mlir::Type type);

}  // namespace vpux
