//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/check_shrink_matmul_groups.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"

namespace vpux {
namespace IE {

bool checkMatMul(IE::MatMulOp origOp) {
    const auto is4DShape = [](ShapeRef shape) {
        return shape.size() == 4;
    };
    auto lhs = origOp.getInput1();
    auto rhs = origOp.getInput2();

    auto lhsShape = getShape(lhs);
    auto rhsShape = getShape(rhs);

    if (!is4DShape(lhsShape) || !is4DShape(rhsShape)) {
        return false;
    }

    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    // Right now it's expected to be the case when transposeA = false and transposeB = true
    if (!IE::isMatmulWithRHSTransposition(origOp)) {
        return false;
    }
    if (lhsShape[N] != rhsShape[N] || lhsShape[C] != rhsShape[C] || lhsShape[W] != rhsShape[W]) {
        return false;
    }

    if (lhsShape[H] == 1) {
        return true;
    }

    // The optimization will break the VF pattern for MatMul-Add-Softmax-MatMul in LLM, but VF pattern needs
    // unrolled MatMul, not grouped MatMul. So here we check if it is beneficial for group MatMul.
    bool res = isGroupedMatMulBeneficial(origOp, lhsShape, rhsShape);
    return res;
}

bool checkTranspose(IE::TransposeOp transposeOp) {
    // TransposeOp should only transpose 4D spatial dims
    const auto transposePerm = DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
    return transposePerm == DimsOrder::NCWH;
}

bool checkAffineReshape(IE::AffineReshapeOp affineReshapeOp) {
    if (affineReshapeOp == nullptr) {
        return false;
    }

    auto inputShape = getShape(affineReshapeOp.getInput());
    auto outputShape = getShape(affineReshapeOp.getOutput());

    // ensure the input/output shape of AffineReshapeOp to be 5/4D
    if (inputShape.size() != 5 || outputShape.size() != 4) {
        return false;
    }

    // Since Matmul actually happens on 2D tensors, we should only check the last 2 dims
    return (inputShape[Dims5D::Act::H] == outputShape[Dims4D::Act::H] &&
            inputShape[Dims5D::Act::W] == outputShape[Dims4D::Act::W]);
}

bool checkBroadCast(IE::BroadcastOp broadcastOp) {
    if (broadcastOp == nullptr) {
        return false;
    }

    const auto is5DShape = [](ShapeRef shape) {
        return shape.size() == 5;
    };
    auto inputShape = getShape(broadcastOp.getInput());
    auto outputShape = getShape(broadcastOp.getOutput());
    if (!is5DShape(inputShape) || !is5DShape(outputShape)) {
        return false;
    }

    auto broadCastDim = IE::getDiffInOutSizeDims(inputShape, outputShape);
    if (broadCastDim.size() != 1) {
        return false;
    }

    // BroadcastOp should broadcast 5D tensor on the first spatial dim (d2)
    return broadCastDim.front() == Dims5D::Act::getSpatialDim(0) && inputShape[broadCastDim.front()] == 1;
}

bool shouldShrinkMatmulGroups(IE::MatMulOp matmulOp) {
    auto rhs = matmulOp.getInput2();
    if (!checkMatMul(matmulOp)) {
        return false;
    }
    IE::AffineReshapeOp reshapeOp = nullptr;
    auto transposeOp = rhs.getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr) {
        reshapeOp = rhs.getDefiningOp<IE::AffineReshapeOp>();
    } else {
        if (!checkTranspose(transposeOp)) {
            return false;
        }
        reshapeOp = transposeOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    }

    if (!checkAffineReshape(reshapeOp)) {
        return false;
    }

    auto broadCastOp = reshapeOp.getInput().getDefiningOp<IE::BroadcastOp>();
    if (!checkBroadCast(broadCastOp)) {
        return false;
    }
    return true;
}

}  // namespace IE
}  // namespace vpux
