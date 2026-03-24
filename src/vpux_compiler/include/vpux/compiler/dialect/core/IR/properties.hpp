//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/IR/Attributes.h>

#include <cstdint>

namespace vpux {

//
// SmallVector<Integer> property
//
template <typename Integer>
mlir::LogicalResult convertFromAttribute(SmallVector<Integer>& storage, mlir::Attribute attr,
                                         llvm::function_ref<mlir::InFlightDiagnostic()>) {
    auto arrayAttr = llvm::dyn_cast_if_present<::mlir::ArrayAttr>(attr);
    if (!arrayAttr) {
        return mlir::failure();
    }
    storage = parseIntArrayAttr<Integer>(arrayAttr);
    return mlir::success();
}

template <typename Integer>
mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, const SmallVector<Integer>& storage) {
    return getIntArrayAttr(ctx, storage);
}

template <typename Integer>
llvm::hash_code hash_value(ArrayRef<Integer> storage) {
    return llvm::hash_value(storage);
}

}  // namespace vpux
