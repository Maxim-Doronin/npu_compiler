//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <functional>

namespace std {

template <>
struct hash<mlir::Attribute> final {
    size_t operator()(mlir::Attribute obj) const {
        return mlir::hash_value(obj);
    }
};

template <>
struct hash<mlir::Type> final {
    size_t operator()(mlir::Type obj) const {
        return mlir::hash_value(obj);
    }
};

}  // namespace std

namespace vpux {

template <typename... Args>
void hashOptionalContents(llvm::hash_code& opHash, Args&&... args) {
    auto processArg = [&opHash](auto&& arg) {
        if (arg.has_value()) {
            opHash = llvm::hash_combine(opHash, llvm::hash_value(llvm::formatv("{0}", arg.value()).str()));
        }
    };
    (processArg(std::forward<Args>(args)), ...);
}

// Hash operation based upon their operation name, attributes, operand types and result types.
llvm::hash_code hashOperation(mlir::Operation* op);

llvm::hash_code hashOperationForTiling(mlir::Operation* op);

llvm::hash_code hashOperationForTilingExcludingAttr(mlir::Operation* op, mlir::StringRef excludedAttrName);

}  // namespace vpux
