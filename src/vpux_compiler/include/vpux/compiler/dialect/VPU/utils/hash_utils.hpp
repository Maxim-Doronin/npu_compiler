//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Operation.h>

namespace vpux::VPU {

template <typename... Args>
void hashOptionalContents(llvm::hash_code& opHash, Args&&... args) {
    auto processArg = [&opHash](auto&& arg) {
        if (arg.has_value()) {
            opHash = llvm::hash_combine(opHash, llvm::hash_value(llvm::formatv("{0}", arg.value()).str()));
        }
    };
    (processArg(std::forward<Args>(args)), ...);
}

llvm::hash_code hashOperationForTiling(mlir::Operation* op);

llvm::hash_code hashOperationForTilingExcludingAttr(mlir::Operation* op, mlir::StringRef excludedAttrName);

llvm::hash_code hashOperationWithCustomAttr(mlir::Operation* op, mlir::StringRef customAttrName,
                                            mlir::Attribute customAttrValue);

}  // namespace vpux::VPU
