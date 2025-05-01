//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MOVEMULTIPLYPOSTOP
#define GEN_PASS_DEF_MOVEMULTIPLYPOSTOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MoveMultiplyPostMatmul
//

//                  (1x32x1024x80)           (1x1x1x1)
//                              \               /
//      (1x32x1x80)         IE.Multiply (1x32x1024x80)
//               \            /
//            IE.Matmul(1x32x1x1024)

// To

//        (1x32x1x80)    1x32x1024x80)
//              \            /
//             IE.Matmul(1x32x1x1024)       (1x1x1x1)
//                         \                   /
//                        IE.Multiply (1x32x1x1024)

template <typename ConcreteOp>
class MoveMultiplyPostLayerGeneric final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MoveMultiplyPostLayerGeneric(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("MoveMultiplyPostLayerGeneric");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalTransformation(ConcreteOp op) const;
    Logger _log;
};

template <typename ConcreteOp>
bool MoveMultiplyPostLayerGeneric<ConcreteOp>::isLegalTransformation(ConcreteOp op) const {
    // IE::MatMulOp should not have post op
    if constexpr (std::is_same_v<ConcreteOp, IE::MatMulOp>) {
        return op.getPostOpAttr() == nullptr;
    }
    // IE::FullyConnectedOp should not have bias
    else if constexpr (std::is_same_v<ConcreteOp, IE::FullyConnectedOp>) {
        return op.getBias() == nullptr;
    }

    return false;
}

bool isBeneficialToConvert(ShapeRef inShape, ShapeRef outShape) {
    return inShape.totalSize() > outShape.totalSize();
}

mlir::Value getSingleDataInput(IE::MultiplyOp multiplyOp) {
    for (auto operand : multiplyOp->getOperands()) {
        if (auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(operand.getDefiningOp())) {
            if (IE::isBaseContentSplat(constOp)) {
                return operand;
            }
            return nullptr;
        }
        auto shape = getShape(operand);
        if (vpux::details::calcTotalShapeSize(shape.raw()) == 1) {
            return operand;
        }
    }
    return nullptr;
}

template <typename ConcreteOp>
mlir::LogicalResult MoveMultiplyPostLayerGeneric<ConcreteOp>::matchAndRewrite(IE::MultiplyOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got multiply layer at '{1}'", origOp->getName(), origOp->getLoc());
    if (!origOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "multiply has more than one user");
    }

    if (origOp.getPostOpAttr() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "multiply has post op attr");
    }

    if (origOp.getClampAttr() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "multiply has clamp attr");
    }

    auto singleDataInput = getSingleDataInput(origOp);
    if (singleDataInput == nullptr) {
        return matchFailed(_log, rewriter, origOp, "multiply doesn't have single data input");
    }

    const mlir::Value nonSingleDataOperand =
            origOp.getInput1() == singleDataInput ? origOp.getInput2() : origOp.getInput1();

    auto layerOp = mlir::dyn_cast<ConcreteOp>(*origOp.getOutput().getUsers().begin());
    if (layerOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "invalid multiply user");
    }

    if (!isLegalTransformation(layerOp)) {
        return matchFailed(_log, rewriter, origOp, "illegal to swap multiply with layerOp");
    }

    if (!isBeneficialToConvert(getShape(origOp.getOutput()), getShape(layerOp.getOutput()))) {
        return matchFailed(_log, rewriter, origOp, "not benefical to swap multiply with layerOp");
    }

    rewriter.setInsertionPoint(layerOp);
    auto origLhs = layerOp->getOperand(0);
    auto origRhs = layerOp->getOperand(1);
    mlir::Value newLhs = origLhs.getDefiningOp() == origOp ? nonSingleDataOperand : origLhs;
    mlir::Value newRhs = origRhs.getDefiningOp() == origOp ? nonSingleDataOperand : origRhs;
    SmallVector<mlir::Value> newOperands = {newLhs, newRhs};
    mlir::IRMapping mapper;
    mapper.map(layerOp->getOperands(), newOperands);
    auto newLayerOp = rewriter.clone(*layerOp, mapper);
    mlir::Value newLayerOpOutput = newLayerOp->getResult(0);

    auto multiplyInput1 = origOp.getInput1() == singleDataInput ? origOp.getInput1() : newLayerOpOutput;
    auto multiplyInput2 = origOp.getInput2() == singleDataInput ? origOp.getInput2() : newLayerOpOutput;

    auto newMultiply = rewriter.create<IE::MultiplyOp>(
            origOp->getLoc(), multiplyInput1, multiplyInput2, origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    rewriter.replaceOp(layerOp, newMultiply.getOutput());
    _log.trace("Successfully swap multiply with layerOp");

    return mlir::success();
}

class MoveMultiplyPostConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MoveMultiplyPostConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("MoveMultiplyPostConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Reshape from 32x64 to 1x32x64
bool isUnsqueezeLikeReshape(IE::ReshapeOp reshapeOp) {
    SmallVector<int64_t> inShape(getShape(reshapeOp.getInput()).raw());
    SmallVector<int64_t> outShape(getShape(reshapeOp.getOutput()).raw());
    if (outShape.size() <= inShape.size()) {
        return false;
    }
    outShape.erase(outShape.begin(), outShape.begin() + outShape.size() - inShape.size());

    return inShape == outShape;
}

bool isOptimizableMultiplyOp(IE::MultiplyOp multiplyOp) {
    auto leftInShape = getShape(multiplyOp.getInput1());
    auto rightInShape = getShape(multiplyOp.getInput2());
    auto outShape = getShape(multiplyOp.getOutput());
    if (leftInShape != outShape || rightInShape != outShape) {
        return false;
    }

    if (multiplyOp.getPostOpAttr() != nullptr || multiplyOp.getClampAttr() != nullptr ||
        multiplyOp.getOutputPaddingAttr() != nullptr || multiplyOp.getInputPaddingAttr() != nullptr) {
        return false;
    }

    // In LLM, the optimization is only for KVcache, still need to keep multiply after FC/Matmul
    // for prefill to get better performance.
    return outShape.raw().front() == 1;
}

mlir::LogicalResult MoveMultiplyPostConcat::matchAndRewrite(IE::ConcatOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Concat layer at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = origOp.getContext();
    // if concat doesn't have static offst attr, then it is single axis concat
    if (origOp.getStaticOffsetsAttr() != nullptr) {
        auto axis = IE::getConcatModifiedAxis(origOp);
        if (axis.size() > 1) {
            return matchFailed(rewriter, origOp, "concat has multi axis");
        }
    }

    SmallVector<IE::MultiplyOp> multiplyOps;
    SmallVector<IE::ReshapeOp> reshapeOps;
    auto inputNums = origOp.getOperands().size();
    for (auto input : origOp.getOperands()) {
        if (auto reshapeOp = mlir::dyn_cast_or_null<IE::ReshapeOp>(input.getDefiningOp())) {
            if (reshapeOp->hasOneUse() && isUnsqueezeLikeReshape(reshapeOp)) {
                if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(reshapeOp.getInput().getDefiningOp())) {
                    reshapeOps.push_back(reshapeOp);
                    multiplyOps.push_back(multiplyOp);
                }
            }
        }
        if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(input.getDefiningOp())) {
            multiplyOps.push_back(multiplyOp);
        }
    }

    if (multiplyOps.size() != inputNums || (reshapeOps.size() != 0 && reshapeOps.size() != inputNums)) {
        return matchFailed(rewriter, origOp, "not all of concat parent is multiply");
    }

    SmallVector<mlir::Value> multiplyLeftInputs;
    SmallVector<mlir::Value> multiplyRightInputs;
    for (auto multiplyOp : multiplyOps) {
        if (!isOptimizableMultiplyOp(multiplyOp)) {
            return matchFailed(rewriter, origOp, "not optimizable multiply");
        }
        multiplyLeftInputs.push_back(multiplyOp.getInput1());
        multiplyRightInputs.push_back(multiplyOp.getInput2());
    }

    if (reshapeOps.size() == inputNums) {
        multiplyLeftInputs.clear();
        multiplyRightInputs.clear();
        for (auto p : multiplyOps | indexed) {
            auto multiplyOp = p.value();
            auto reshapeOp = reshapeOps[p.index()];
            auto leftReshape =
                    rewriter.create<IE::ReshapeOp>(appendLoc(reshapeOp->getLoc(), "_left_reshape"),
                                                   multiplyOp.getInput1(), nullptr, false,
                                                   getIntArrayAttr(ctx, getShape(reshapeOp.getOutput()).raw()))
                            .getOutput();
            multiplyLeftInputs.push_back(leftReshape);

            auto rightReshape =
                    rewriter.create<IE::ReshapeOp>(appendLoc(reshapeOp->getLoc(), "_right_reshape"),
                                                   multiplyOp.getInput2(), nullptr, false,
                                                   getIntArrayAttr(ctx, getShape(reshapeOp.getOutput()).raw()))
                            .getOutput();
            multiplyRightInputs.push_back(rightReshape);
        }
    }

    auto multiplyRightConcat = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_right_concat"),
                                                             mlir::ValueRange(multiplyRightInputs),
                                                             origOp.getPerAxisAttr(), origOp.getStaticOffsetsAttr());
    auto multiplyLeftConcat = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_left_concat"),
                                                            mlir::ValueRange(multiplyLeftInputs),
                                                            origOp.getPerAxisAttr(), origOp.getStaticOffsetsAttr());
    auto multiply = rewriter.create<IE::MultiplyOp>(appendLoc(multiplyOps.front()->getLoc(), "_multiply_after_concat"),
                                                    multiplyLeftConcat.getOutput(), multiplyRightConcat.getOutput(),
                                                    multiplyOps.front().getAutoBroadcastAttr(), nullptr, nullptr,
                                                    nullptr, nullptr);
    rewriter.replaceOp(origOp, multiply.getOutput());

    _log.trace("Successfully move multiply post Concat");
    return mlir::success();
}

//
// MoveMultiplyPostOpPass
//

class MoveMultiplyPostOpPass final : public IE::impl::MoveMultiplyPostOpBase<MoveMultiplyPostOpPass> {
public:
    explicit MoveMultiplyPostOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MoveMultiplyPostOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveMultiplyPostLayerGeneric<IE::MatMulOp>>(&ctx, _log);
    patterns.add<MoveMultiplyPostLayerGeneric<IE::FullyConnectedOp>>(&ctx, _log);
    patterns.add<MoveMultiplyPostConcat>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveMultiplyPostOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMoveMultiplyPostOpPass(Logger log) {
    return std::make_unique<MoveMultiplyPostOpPass>(log);
}
