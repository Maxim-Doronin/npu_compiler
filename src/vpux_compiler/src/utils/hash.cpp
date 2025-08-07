//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/hash.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/utils/types.hpp"

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
};
}  // namespace

llvm::hash_code vpux::hashOperation(mlir::Operation* op) {
    auto hash = llvm::hash_combine(op->getName(), op->getDiscardableAttrDictionary(), op->getResultTypes(),
                                   op->hashProperties());
    for (auto operand : op->getOperands()) {
        hash = llvm::hash_combine(hash, mlir::hash_value(operand.getType()));
    }
    return hash;
}

llvm::hash_code vpux::hashOperationForTiling(mlir::Operation* op) {
    llvm::hash_code hash = llvm::hash_combine(op->getName(), op->getRawDictionaryAttrs(), op->hashProperties());
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

llvm::hash_code vpux::hashOperationForTilingExcludingAttr(mlir::Operation* op, mlir::StringRef excludedAttrName) {
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
