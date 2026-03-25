//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
namespace IE {

namespace details {
template <typename Builder>
mlir::Operation* cloneMatMulOp(Builder& builder, IE::MatMulOp matMulOp, mlir::Value newInput1, mlir::Value newInput2) {
    mlir::IRMapping mapper;
    mapper.map(matMulOp.getInput1(), newInput1);
    mapper.map(matMulOp.getInput2(), newInput2);

    auto* newOp = builder.clone(*matMulOp.getOperation(), mapper);
    return newOp;
}
}  // namespace details

/** Clone a MatMul operation with new inputs.
 *
 *  This function clones the given MatMul operation, replacing its inputs with the provided new inputs.
 *
 *  Please note that rest of attributes (like transposition flags, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
mlir::Operation* cloneMatMulOp(Builder& builder, IE::MatMulOp matMulOp, mlir::Value newInput1, mlir::Value newInput2) {
    auto* newOp = details::cloneMatMulOp(builder, matMulOp, newInput1, newInput2);

    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);
    return newOp;
}

/** Clone a MatMul operation with new inputs and transposition flags.
 *  This function clones the given MatMul operation, replacing its inputs with the provided new inputs
 *  and setting the transposition flags.
 *
 *  Please note that rest of attributes (like post-ops, clam, etc.) are preserved from the original operation.
 */
template <typename Builder>
mlir::Operation* cloneMatMulOp(Builder& builder, IE::MatMulOp matMulOp, mlir::Value newInput1, mlir::Value newInput2,
                               bool transposeA, bool transposeB) {
    auto* newOp = details::cloneMatMulOp(builder, matMulOp, newInput1, newInput2);
    auto newMatMulOp = mlir::cast<IE::MatMulOp>(*newOp);
    newMatMulOp.setTransposeA(transposeA);
    newMatMulOp.setTransposeB(transposeB);

    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);
    return newOp;
}

/** Clone a MatMul operation with new inputs and output type.
 *
 *  This function clones the given MatMul operation, replacing its inputs with the provided new inputs
 *  and setting the output type.
 *
 *  Please note that rest of attributes (like transposition flags, post-ops, etc.) are preserved from the original
 * operation.
 */
template <typename Builder>
mlir::Operation* cloneMatMulOp(Builder& builder, IE::MatMulOp matMulOp, mlir::Type outputType, mlir::Value newInput1,
                               mlir::Value newInput2) {
    auto newMatMulOp = details::cloneMatMulOp(builder, matMulOp, newInput1, newInput2);
    newMatMulOp->getResult(0).setType(outputType);

    return newMatMulOp;
}

// E#154850: This function will/must be removed when regressions are addressed with tiling specific subgraphs
bool isGroupedMatMulBeneficial(IE::MatMulOp matmulOp, ShapeRef input1Shape, ShapeRef input2Shape);

bool isGroupedMatMulBeneficialToGroupConv(IE::MatMulOp matmulOp);

bool isMatmulWithRHSTransposition(IE::MatMulOp matmulOp);

}  // namespace IE
}  // namespace vpux
