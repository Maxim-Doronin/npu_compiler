//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/jit_utils.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>

namespace vpux::ShaveCodeGen {

//
// ShaveCodeGenSupportedOpInterface
//

namespace {
// The predicates below are intentionally conservative:
//  * Quantized types: rejected per-axis quant.
//  * bfloat16 element types: rejected.
//  * Any shaped type with sparse encoding and dynamic: rejected.
//  * Otherwise, scalars and shaped types are accepted if their element type is.

/// Returns true if the *scalar* element type is supported.
bool isSupportedElementType(mlir::Type type) {
    // Reject per-axis quant. Accept per-tensor quant
    if (auto st = mlir::dyn_cast<mlir::ShapedType>(type)) {
        type = st.getElementType();
    }
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(type)) {
        return false;
    }
    if (mlir::isa<mlir::quant::QuantizedType>(type)) {
        return mlir::isa<mlir::quant::UniformQuantizedType>(type);
    }

    // Reject bfloat16 element types.
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(type)) {
        return !ft.isBF16();
    }

    // All other element kinds (integers, index, other floats, etc.) are OK.
    return true;
}

/// Returns true if and only if the shaped type is supported under current rules.
/// Rejects any sparse-encoded shaped type, otherwise delegates to the element-type predicate.
bool isSupportedShapedType(mlir::ShapedType st) {
    // Disallow MLIR sparse encodings.
    if (mlir::sparse_tensor::getSparseTensorEncoding(st)) {
        return false;
    }

    // Require rank and fully static shape.
    if (!st.hasRank() || !st.hasStaticShape()) {
        return false;
    }

    return isSupportedElementType(st.getElementType());
}

/// Returns true if the (possibly shaped) type is supported.
bool isSupportedType(mlir::Type type) {
    if (auto st = mlir::dyn_cast<mlir::ShapedType>(type)) {
        return isSupportedShapedType(st);
    }
    return isSupportedElementType(type);
}
}  // namespace

bool hasOnlySupportedTypes(mlir::Operation* op) {
    // Check operands.
    bool operandsOK = llvm::all_of(op->getOperands(), [](mlir::Value v) {
        return isSupportedType(v.getType());
    });

    if (!operandsOK) {
        return false;
    }

    // Check results.
    return llvm::all_of(op->getResultTypes(), [](mlir::Type t) {
        return isSupportedType(t);
    });
}

}  // namespace vpux::ShaveCodeGen
