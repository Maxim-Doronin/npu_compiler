//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/check_shrink_matmul_groups.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SHRINKMATMULGROUPS
#define GEN_PASS_DEF_SHRINKMATMULGROUPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ShrinkMatmulGroups
//

/*

Case 1:
    Convert below 24 groups Matmul:
                        RHS
                    1x8x1x1024x64
                        |
                    Broadcast
                        |
                    1x8x3x1024x64
                        |
                    AffineReshape
        LHS             |
    1x24x1x64       1x24x1024x64
        \               /
             MatMul

    to a new 8 groups Matmul:
        LHS             RHS
    1x24x1x64       1x8x1x1024x64
        |               |
    Reshape         Reshape
        |               |
    1x8x3x64        1x8x1024x64
        \               /
            MatMul

Case 2:
    Convert below 24 groups Matmul:

                        RHS
                    1x8x1x1024x64
                        |
                    Broadcast
                        |
                    1x8x3x1024x64
                        |
                    AffineReshape
                        |
                    1x24x1024x64
                        |
                    Transpose
        LHS             |
    1x24x1x1024     1x24x64x1024
        \               /
             MatMul

    to a new 8 groups Matmul:

                        RHS
                    1x8x1x1024x64
                        |
                    Reshape
        LHS             |
    1x24x1x64       1x8x1024x64
        |               |
    Reshape         Transpose
        |               |
    1x8x3x64        1x24x64x1024
        \                /
            MatMul
*/

class ShrinkMatmulGroups final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    ShrinkMatmulGroups(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
        setDebugName("ShrinkMatmulGroups");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ShrinkMatmulGroups::matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MatMulOp at '{1}'", origOp->getName(), origOp->getLoc());

    auto lhs = origOp.getInput1();
    auto rhs = origOp.getInput2();

    if (!IE::checkMatMul(origOp)) {
        return mlir::failure();
    }
    IE::AffineReshapeOp reshapeOp = nullptr;
    auto transposeOp = rhs.getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr) {
        reshapeOp = rhs.getDefiningOp<IE::AffineReshapeOp>();
    } else {
        if (!IE::checkTranspose(transposeOp)) {
            return mlir::failure();
        }
        reshapeOp = transposeOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    }

    if (!IE::checkAffineReshape(reshapeOp)) {
        return mlir::failure();
    }

    auto broadCastOp = reshapeOp.getInput().getDefiningOp<IE::BroadcastOp>();
    if (!IE::checkBroadCast(broadCastOp)) {
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();
    auto broadcastOutputShape = getShape(broadCastOp.getOutput());
    int64_t newGroupNum = broadcastOutputShape[Dims5D::Act::C];

    // Create new LHS by reshaping the original LHS
    auto origLhsShape = getShape(lhs);
    SmallVector<int64_t> lhsTargetShape = to_small_vector(origLhsShape);
    lhsTargetShape[Dims4D::Act::C.ind()] = newGroupNum;
    lhsTargetShape[Dims4D::Act::H.ind()] = origLhsShape[Dims4D::Act::H] * origLhsShape[Dims4D::Act::C] / newGroupNum;
    VPUX_THROW_WHEN(origLhsShape[Dims4D::Act::C] % newGroupNum != 0, "Unexpected origLhsShape {0} and newGroupNum {1}",
                    origLhsShape, newGroupNum);
    const auto lhsTargetShapeAttr = getIntArrayAttr(ctx, lhsTargetShape);
    auto newLhs = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "lhs_reshape"), lhs, nullptr, false,
                                                 lhsTargetShapeAttr)
                          .getOutput();

    // Create new RHS chain: reshape and transpose the original input of BroadCastOp
    SmallVector<int64_t> targetShape = to_small_vector(getShape(reshapeOp.getOutput()));
    targetShape[Dims4D::Act::C.ind()] = newGroupNum;
    const auto targetShapeAttr = getIntArrayAttr(ctx, targetShape);
    auto newRhs = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "rhs_reshape"), broadCastOp.getInput(),
                                                 nullptr, false, targetShapeAttr)
                          .getOutput();

    if (transposeOp != nullptr) {
        newRhs = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "rhs_transpose"), newRhs, nullptr,
                                                  transposeOp.getOrderValueAttr())
                         .getOutput();
    }

    // Create new group Matmul
    auto newMatMul = cloneMatMulOp(rewriter, origOp, newLhs, newRhs);
    newMatMul->setLoc(appendLoc(origOp->getLoc(), "new_group_mul"));

    auto outputShape = getShape(origOp.getOutput());
    const auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
    auto outReshape = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "output_reshape"),
                                                     newMatMul->getResult(0), nullptr, false, outputShapeAttr);

    _log.trace("Successfully shrunk number of groups at {0}", origOp.getLoc());
    rewriter.replaceOp(origOp, outReshape.getOutput());

    return mlir::success();
}

//
// ShrinkMatmulGroupsPass
//

class ShrinkMatmulGroupsPass final : public IE::impl::ShrinkMatmulGroupsBase<ShrinkMatmulGroupsPass> {
public:
    explicit ShrinkMatmulGroupsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ShrinkMatmulGroupsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ShrinkMatmulGroups>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}
}  // namespace

//
// createShrinkMatmulGroupsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createShrinkMatmulGroupsPass(Logger log) {
    return std::make_unique<ShrinkMatmulGroupsPass>(log);
}
