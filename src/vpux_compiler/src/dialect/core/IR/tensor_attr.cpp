//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <utility>

using namespace vpux;

constexpr StringLiteral orderName = "order";
constexpr StringLiteral memSpaceName = "mem_space";
constexpr StringLiteral boundsName = "bounds";
constexpr StringLiteral dynamicDimsMaskName = "dynamic_dims_mask";

template <class T>
bool checkAttr(mlir::DictionaryAttr derived, StringRef attrName, int& numAbsentAttrs) {
    auto attr = derived.get(attrName);
    if (attr == nullptr) {
        ++numAbsentAttrs;
        return true;
    }

    return attr.isa<T>();
}

bool vpux::TensorAttr::classof(mlir::Attribute attr) {
    if (attr == nullptr) {
        return false;
    }

    auto derived = attr.dyn_cast<mlir::DictionaryAttr>();
    if (derived == nullptr) {
        return false;
    }

    int numAbsentAttrs = 0;

    if (!checkAttr<mlir::AffineMapAttr>(derived, orderName, numAbsentAttrs)) {
        return false;
    }
    if (!checkAttr<vpux::IndexedSymbolAttr>(derived, memSpaceName, numAbsentAttrs)) {
        return false;
    }
    if (!checkAttr<Const::OpaqueI64ElementsAttr>(derived, boundsName, numAbsentAttrs)) {
        return false;
    }
    if (!checkAttr<Const::OpaqueI64ElementsAttr>(derived, dynamicDimsMaskName, numAbsentAttrs)) {
        return false;
    }

    return (derived.size() + numAbsentAttrs) == 4;
}

Const::OpaqueI64ElementsAttr createOpaqueI64ElementsAttr(mlir::MLIRContext* context, ArrayRef<int64_t> bounds) {
    const auto elemType = mlir::IntegerType::get(context, 64, mlir::IntegerType::Signed);
    const auto dataStorageType = mlir::RankedTensorType::get({checked_cast<int64_t>(bounds.size())}, elemType);
    return Const::OpaqueI64ElementsAttr::get(dataStorageType, bounds);
}

TensorAttr vpux::TensorAttr::get(mlir::MLIRContext* context, mlir::AffineMapAttr order,
                                 vpux::IndexedSymbolAttr memSpace, Bounds bounds, DynamicDimsMask dynamicDimsMask) {
    VPUX_THROW_WHEN(!bounds.empty() && !dynamicDimsMask.empty(),
                    "Ambiguous tensor representation. Both Bounds {0} and DynamicDimsMask {1} were provided.", bounds,
                    dynamicDimsMask);

    SmallVector<mlir::NamedAttribute> fields;

    if (order != nullptr) {
        auto orderId = mlir::StringAttr::get(context, orderName);
        fields.emplace_back(orderId, order);
    }

    if (memSpace != nullptr) {
        auto memSpaceId = mlir::StringAttr::get(context, memSpaceName);
        fields.emplace_back(memSpaceId, memSpace);
    }

    if (!bounds.empty()) {
        auto boundsId = mlir::StringAttr::get(context, boundsName);
        const auto opaqueAttr = createOpaqueI64ElementsAttr(context, bounds.raw());
        fields.emplace_back(boundsId, opaqueAttr);
    }

    if (!dynamicDimsMask.empty()) {
        bool onlyZeroOrOne = llvm::all_of(dynamicDimsMask, [](auto value) {
            return value == 0 || value == 1;
        });
        VPUX_THROW_UNLESS(onlyZeroOrOne, "Dynamic dims mask must have only 0 or 1, got {0}", dynamicDimsMask);

        bool allZeroes = llvm::all_of(dynamicDimsMask, [](auto value) {
            return value == 0;
        });
        VPUX_THROW_UNLESS(!allZeroes, "Dynamic dims mask must contain 1's that represent dynamic dimensions, got {0}",
                          dynamicDimsMask);

        auto dynamicDimsMaskId = mlir::StringAttr::get(context, dynamicDimsMaskName);
        const auto opaqueAttr = createOpaqueI64ElementsAttr(context, dynamicDimsMask.raw());
        fields.emplace_back(dynamicDimsMaskId, opaqueAttr);
    }

    auto dict = mlir::DictionaryAttr::get(context, fields);
    return dict.dyn_cast<TensorAttr>();
}

template <class T>
T getAttr(mlir::DictionaryAttr derived, StringRef attrName) {
    auto attr = derived.get(attrName);
    if (attr == nullptr) {
        return nullptr;
    }
    VPUX_THROW_WHEN(!attr.isa<T>(), "incorrect {0} Attribute type found: {1}", attrName, attr);
    return attr.cast<T>();
}

mlir::AffineMapAttr TensorAttr::getOrder() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    return getAttr<mlir::AffineMapAttr>(derived, orderName);
}

vpux::IndexedSymbolAttr TensorAttr::getMemSpace() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    return getAttr<vpux::IndexedSymbolAttr>(derived, memSpaceName);
}

Bounds TensorAttr::getBounds() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    auto bounds = getAttr<Const::OpaqueI64ElementsAttr>(derived, boundsName);
    if (bounds != nullptr) {
        return Bounds(bounds.getValue());
    }

    return Bounds();
}

DynamicDimsMask TensorAttr::getDynamicDimsMask() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    auto dynamicDimsMask = getAttr<Const::OpaqueI64ElementsAttr>(derived, dynamicDimsMaskName);
    if (dynamicDimsMask != nullptr) {
        return DynamicDimsMask(dynamicDimsMask.getValue());
    }

    return DynamicDimsMask();
}

//
// Helpers
//

//
// Default getTensorAttr
//

TensorAttr vpux::getTensorAttr(mlir::MLIRContext* ctx, mlir::AffineMapAttr order, IndexedSymbolAttr memSpace,
                               Bounds bounds, DynamicDimsMask dynamicDimsMask) {
    // Initially, tensors do not have an encoding attribute, which is equivalent to an empty TensorAttr.
    // But in fact, such tensors have a different type: `tensor<1x8x4x2xf16> != tensor<1x8x4x2xf16, {}>`.
    // So let's not use empty attributes to avoid ambiguous representation of the same type.
    if ((order == nullptr || order.getValue().isIdentity()) && memSpace == nullptr && bounds.raw().empty() &&
        dynamicDimsMask.raw().empty()) {
        return nullptr;
    }

    return TensorAttr::get(ctx, order, memSpace, std::move(bounds), std::move(dynamicDimsMask));
}

TensorAttr vpux::getTensorAttr(mlir::MLIRContext* ctx, mlir::AffineMap order, IndexedSymbolAttr memSpace, Bounds bounds,
                               DynamicDimsMask dynamicDimsMask) {
    return vpux::getTensorAttr(ctx, mlir::AffineMapAttr::get(order), memSpace, std::move(bounds),
                               std::move(dynamicDimsMask));
}

TensorAttr vpux::getTensorAttr(mlir::MLIRContext* ctx, DimsOrder order, IndexedSymbolAttr memSpace, Bounds bounds,
                               DynamicDimsMask dynamicDimsMask) {
    return vpux::getTensorAttr(ctx, order.toAffineMap(ctx), memSpace, std::move(bounds), std::move(dynamicDimsMask));
}

TensorAttr vpux::getTensorAttr(mlir::RankedTensorType type) {
    if (const auto encoding = type.getEncoding()) {
        const auto tensorAttr = encoding.dyn_cast<TensorAttr>();
        VPUX_THROW_UNLESS(tensorAttr != nullptr, "Unsupported tensor encoding attribute '{0}'", encoding);
        return tensorAttr;
    }

    return nullptr;
}

TensorAttr vpux::getTensorAttr(mlir::RankedTensorType type, mlir::AffineMap order, IndexedSymbolAttr memSpace,
                               mlir::ArrayRef<int64_t> dynamicAttr) {
    if (mlir::isa<Core::BoundedTensorType>(type)) {
        return vpux::getTensorAttr(type.getContext(), order, memSpace, Bounds(dynamicAttr));
    }
    if (mlir::isa<Core::DynamicDimsMaskTensorType>(type)) {
        return vpux::getTensorAttr(type.getContext(), order, memSpace, Bounds(), DynamicDimsMask(dynamicAttr));
    }
    return vpux::getTensorAttr(type.getContext(), order, memSpace);
}

mlir::AffineMap vpux::getOrder(mlir::RankedTensorType type) {
    if (const auto desc = vpux::getTensorAttr(type)) {
        if (const auto orderAttr = desc.getOrder()) {
            return orderAttr.getValue();
        }
    }

    const auto numDims = checked_cast<uint32_t>(type.getRank());
    return mlir::AffineMap::getMinorIdentityMap(numDims, numDims, type.getContext());
}

IndexedSymbolAttr vpux::getMemorySpace(mlir::RankedTensorType type) {
    if (const auto desc = vpux::getTensorAttr(type)) {
        return desc.getMemSpace();
    }

    return nullptr;
}

Bounds vpux::getBounds(mlir::Type type) {
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        return boundedType.getBounds();
    }

    return {};
};

DynamicDimsMask vpux::getDynamicDimsMask(mlir::Type type) {
    if (auto dynamicDimsMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type)) {
        return dynamicDimsMaskType.getDynamicDimsMask();
    }

    return {};
}
