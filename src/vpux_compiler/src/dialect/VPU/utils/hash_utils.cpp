//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/types.hpp"

using namespace vpux;

namespace {
llvm::hash_code hashType(vpux::NDTypeInterface type) {
    auto hash = llvm::hash_combine(type.getTotalAllocSize().count(), type.getElemTypeSize().count());
    hash = llvm::hash_combine(hash, llvm::hash_combine_range(type.getShape().begin(), type.getShape().end()));
    for (auto& stride : type.getStrides()) {
        hash = llvm::hash_combine(hash, stride.count());
    }
    if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
        hash = llvm::hash_combine(hash, sparseType.getData(), sparseType.getSparsityCompression(),
                                  sparseType.getSparsityMap(), sparseType.getStorageElementTable());
    }
    return hash;
}
}  // namespace

namespace vpux::VPU {

llvm::hash_code hashOperationForTiling(mlir::Operation* op) {
    llvm::hash_code hash = llvm::hash_combine(op->getName(), op->getAttrDictionary(), op->hashProperties());
    // When doing tiling, the compiler only considers data size related attr for data type, attr like quant
    // zp or scales are ignored. So related hash calculation should avoid taking those attrs into consideration
    for (auto operand : op->getOperands()) {
        hash = llvm::hash_combine(hash, hashType(mlir::cast<vpux::NDTypeInterface>(operand.getType())));
    }
    for (auto result : op->getResults()) {
        hash = llvm::hash_combine(hash, hashType(mlir::cast<vpux::NDTypeInterface>(result.getType())));
    }
    return hash;
}

llvm::hash_code hashOperationForTilingExcludingAttr(mlir::Operation* op, mlir::StringRef excludedAttrName) {
    if (!op->hasAttr(excludedAttrName)) {
        return hashOperationForTiling(op);
    }
    auto dictAttrs = op->getAttrDictionary();
    SmallVector<mlir::NamedAttribute> filteredAttrs;

    for (auto attr : dictAttrs) {
        if (attr.getName() != excludedAttrName) {
            filteredAttrs.push_back(attr);
        }
    }

    auto opHash = llvm::hash_combine(op->getName(), mlir::DictionaryAttr::get(op->getContext(), filteredAttrs));

    // When doing tiling, the compiler only considers data size related attr for data type, attr like quant
    // zp or scales are ignored. So related hash calculation should avoid taking those attrs into consideration
    for (auto operand : op->getOperands()) {
        opHash = llvm::hash_combine(opHash, hashType(mlir::cast<vpux::NDTypeInterface>(operand.getType())));
    }
    for (auto result : op->getResults()) {
        opHash = llvm::hash_combine(opHash, hashType(mlir::cast<vpux::NDTypeInterface>(result.getType())));
    }
    return opHash;
}

llvm::hash_code hashOperationWithCustomAttr(mlir::Operation* op, mlir::StringRef customAttrName,
                                            mlir::Attribute customAttrValue) {
    auto dictAttrs = op->getAttrDictionary();
    SmallVector<mlir::NamedAttribute> filteredAttrs;

    for (auto attr : dictAttrs) {
        if (attr.getName() != customAttrName) {
            filteredAttrs.push_back(attr);
        }
    }
    filteredAttrs.push_back(mlir::NamedAttribute(customAttrName, customAttrValue));
    auto opHash = llvm::hash_combine(op->getName(), mlir::DictionaryAttr::get(op->getContext(), filteredAttrs),
                                     op->hashProperties());

    // When doing tiling, the compiler only considers data size related attr for data type, attr like quant
    // zp or scales are ignored. So related hash calculation should avoid taking those attrs into consideration
    for (auto operand : op->getOperands()) {
        opHash = llvm::hash_combine(opHash, hashType(mlir::cast<vpux::NDTypeInterface>(operand.getType())));
    }
    for (auto result : op->getResults()) {
        opHash = llvm::hash_combine(opHash, hashType(mlir::cast<vpux::NDTypeInterface>(result.getType())));
    }
    return opHash;
}

}  // namespace vpux::VPU
