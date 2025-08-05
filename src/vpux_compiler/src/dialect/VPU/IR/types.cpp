//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPU/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUDialect::registerTypes
//

void VPU::VPUDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPU/types.cpp.inc>
            >();
}

//
// VPU::DistributedTensorType accessors
//

ShapeRef VPU::DistributedTensorType::getShape() const {
    return ShapeRef(getImpl()->shape);
}

mlir::Type VPU::DistributedTensorType::getElementType() const {
    return getImpl()->elementType;
}

mlir::AffineMapAttr VPU::DistributedTensorType::getOrder() const {
    return getImpl()->order;
}

IndexedSymbolAttr VPU::DistributedTensorType::getMemSpace() const {
    return getImpl()->memSpace;
}

VPU::DistributionInfoAttr VPU::DistributedTensorType::getDistribution() const {
    return getImpl()->distribution;
}

Const::OpaqueI64ElementsAttr VPU::DistributedTensorType::getDynamicDimsMask() const {
    return getImpl()->dynamicDimsMask;
}

VPU::DistributedTensorType VPU::DistributedTensorType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                                 mlir::Type elementType) const {
    if (!shape.has_value()) {
        return mlir::cast<vpux::VPU::DistributedTensorType>(changeElemType(elementType));
    }
    return mlir::cast<vpux::VPU::DistributedTensorType>(changeShapeElemType(ShapeRef(shape.value()), elementType));
}

//
// VPU::SparseTensorType accessors
//

VPU::SparseTensorType VPU::SparseTensorType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                       mlir::Type elementType) const {
    if (!shape.has_value()) {
        return mlir::cast<vpux::VPU::SparseTensorType>(changeElemType(elementType));
    }
    return mlir::cast<vpux::VPU::SparseTensorType>(changeShapeElemType(ShapeRef(shape.value()), elementType));
}
