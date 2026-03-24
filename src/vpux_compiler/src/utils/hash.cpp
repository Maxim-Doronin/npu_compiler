//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/hash.hpp"

#include <mlir/IR/Operation.h>

using namespace vpux;

llvm::hash_code vpux::hashOperation(mlir::Operation* op) {
    auto hash = llvm::hash_combine(op->getName(), op->getDiscardableAttrDictionary(), op->getResultTypes(),
                                   op->hashProperties());
    for (auto operand : op->getOperands()) {
        hash = llvm::hash_combine(hash, mlir::hash_value(operand.getType()));
    }
    return hash;
}
