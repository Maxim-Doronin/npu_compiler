//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

namespace mlir {
class PatternRewriter;
class MLIRContext;
}  // namespace mlir

namespace vpux {
namespace ShaveCodeGen {

/// @brief Converts a ranked tensor to a linalg compatible tensor. The output tensor
/// will have an identity layout. If the element type is a non-signless integer then
/// the element type is changed to be signless. The output tensor has the same
/// memory layout as the input tensor.
/// @param operand - the tensor to convert
/// @param rewriter
/// @return the converted tensor
mlir::Value convertToLinalgValue(mlir::Value operand, mlir::PatternRewriter& rewriter);

/// @brief Converts from a ranked linalg-compatible tensor (identity layout, no non-signless
/// integer element types) to one with the specified output (same memory layout). The specified
/// output type must have the same memory layout as the input tensor.
/// @param operand - the tensor to convert
/// @param outputTy - the output tensor type
/// @return the converted tensor
mlir::Value convertFromLinalgValue(mlir::Value operand, mlir::Type outputTy, mlir::PatternRewriter& rewriter);

/// @brief Get the element type of a linalg-compatible tensor for the input type.
/// @param ty - the input tensor type
/// @param ctx
/// @return the element type
mlir::Type getLinalgElementType(mlir::Type ty, mlir::MLIRContext* ctx);

}  // namespace ShaveCodeGen
}  // namespace vpux
