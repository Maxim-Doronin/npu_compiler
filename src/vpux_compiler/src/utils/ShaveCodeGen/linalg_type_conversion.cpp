//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

namespace vpux {
namespace ShaveCodeGen {

mlir::Type getLinalgElementType(mlir::Type ty, mlir::MLIRContext* ctx) {
    assert(mlir::isa<mlir::RankedTensorType>(ty) && "Ranked tensor type required for getLinalgElementType");
    // Get the math/arith compatible element type for the ty tensor type.
    // math/arith dialects don't accept non-signless integers.
    auto elTy = mlir::cast<NDTypeInterface>(ty).getElementType();
    auto signlessElTy = mlir::isa<mlir::IntegerType>(elTy) && !elTy.isSignlessInteger()
                                ? mlir::IntegerType::get(ctx, getElemTypeSize(elTy).count())
                                : elTy;
    return signlessElTy;
}

mlir::Value convertFromLinalgValue(mlir::Value op, mlir::Type outputTy, mlir::PatternRewriter& rewriter) {
    mlir::Value ret = op;
    auto resultElTy = mlir::cast<NDTypeInterface>(outputTy).getElementType();
    auto elTy = mlir::cast<NDTypeInterface>(op.getType()).getElementType();
    const auto outTy = mlir::cast<vpux::NDTypeInterface>(outputTy);
    bool needsPermute = !outTy.getDimsOrder().isIdentity();
    auto rank = outTy.getRank();

    if (elTy != resultElTy) {
        // Do a tensor.bitcast in order to change the element type.
        auto bcTy = mlir::cast<NDTypeInterface>(ret.getType()).changeElemType(resultElTy);
        ret = rewriter.create<mlir::tensor::BitcastOp>(op.getLoc(), bcTy, ret).getResult();
    }

    if (needsPermute) {
        // Perform a permute cast to get the desired type. This should be a no-op after buffers are allocated.
        ret = rewriter.create<IE::PermuteCastOp>(op.getLoc(), outputTy, ret,
                                                 outTy.getDimsOrder().toAffineMap(rewriter.getContext()),
                                                 rewriter.getMultiDimIdentityMap(rank))
                      ->getResult(0);
    }

    return ret;
}

mlir::Value convertToLinalgValue(mlir::Value operand, mlir::PatternRewriter& rewriter) {
    auto loc = operand.getLoc();
    mlir::Value retVal = operand;
    const auto opTy = mlir::cast<vpux::NDTypeInterface>(operand.getType());
    auto rank = opTy.getRank();
    bool needsPermute = !opTy.getDimsOrder().isIdentity();
    if (needsPermute) {
        // Do a IE.PermuteCast to remove the order.
        auto identMap = rewriter.getMultiDimIdentityMap(rank);
        auto dstShape = Shape(DimsOrder::fromValue(retVal).toMemoryOrder(getShape(retVal)).raw());
        auto retTy = opTy.changeDimsOrder(DimsOrder::fromNumDims(rank)).changeShape(dstShape);
        retVal = rewriter.create<IE::PermuteCastOp>(loc, retTy, retVal, identMap, identMap);
    }

    auto elTy = mlir::cast<NDTypeInterface>(operand.getType()).getElementType();
    auto signlessElTy = getLinalgElementType(operand.getType(), rewriter.getContext());
    if (elTy != signlessElTy) {
        // The input type is a signed/unsigned integer so not compatible
        // with the dialects we need for lowering (math, arith, etc). We need to bitcast this
        // to a signless integer type.
        auto outputTy = mlir::cast<NDTypeInterface>(retVal.getType()).changeElemType(signlessElTy);
        retVal = rewriter.create<mlir::tensor::BitcastOp>(loc, outputTy, retVal)->getResult(0);
    }

    return retVal;
}

}  // namespace ShaveCodeGen
}  // namespace vpux
