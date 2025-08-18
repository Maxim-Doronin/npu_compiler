//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_EXPANDSOFTMAXAXIS
#define GEN_PASS_DEF_EXPANDSOFTMAXAXIS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
#define ALIGNMENT_REQUIREMENT_IN_ELEMENTS 16
namespace {

// The condition for pads is that it should be only over the inner most dimension, at the end
// Softmax will support a maximum of 31 elements padded over inner most dimension
bool arePadsValid(const llvm::SmallVector<int64_t>& padsBegin, const llvm::SmallVector<int64_t>& padsEnd) {
    if (padsBegin.size() != padsEnd.size()) {
        return false;
    }

    size_t rank = padsBegin.size();
    for (size_t i = 0; i < rank - 1; i++) {
        if (padsBegin[i] != 0 || padsEnd[i] != 0) {
            return false;
        }
    }
    if (padsBegin[rank - 1] != 0) {
        return false;
    }
    int64_t paddedDim = padsEnd[rank - 1];
    if (paddedDim < 32) {
        return true;
    }
    return false;
}

//
// SimplifyReshapes
// Remove rehsapes like reshape ([1,N*M,X,Y] -> [N,M,X,Y]) -> SoftMax -> reshape ([N,M,X,Y] -> [1,N*M,X,Y]) and replace
// to
//  [1,N*M,X,Y] -> SoftMax -> [1,N*M,X,Y]

class SimplifyReshapes final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    SimplifyReshapes(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp softMaxOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SimplifyReshapes::matchAndRewrite(IE::SoftMaxOp softMaxOp, mlir::PatternRewriter& rewriter) const {
    // 1. check if softmax input is reshape
    auto inReShapeOp = softMaxOp.getInput().getDefiningOp<IE::ReshapeOp>();
    if (inReShapeOp == nullptr) {
        _log.trace("[{0}] SoftmaxOp '{1}' has no reshape input", getDebugName(), softMaxOp->getName());
        return mlir::failure();
    }
    if (!softMaxOp.getInput().hasOneUse()) {
        _log.trace("[{0}] SoftmaxOp '{1}' has multiple users", getDebugName(), softMaxOp->getName());
        return mlir::failure();
    }

    // 2. check if softmax output is reshape
    if (!softMaxOp.getOutput().hasOneUse()) {
        _log.trace("[{0}] SoftmaxOp '{1}' has multiple users", getDebugName(), softMaxOp->getName());
        return mlir::failure();
    }

    auto outReShapeOp = mlir::dyn_cast<IE::ReshapeOp>(*softMaxOp.getOutput().getUsers().begin());
    if (outReShapeOp == nullptr) {
        _log.trace("[{0}] SoftmaxOp '{1}' has no reshape output", getDebugName(), softMaxOp->getName());
        return mlir::failure();
    }

    // 3. check that inReShapeOp input and outReShapeOp output are the same shape
    auto inReShapeInputType = mlir::cast<vpux::NDTypeInterface>(inReShapeOp.getInput().getType());
    auto outReShapeOutputType = mlir::cast<vpux::NDTypeInterface>(outReShapeOp.getOutput().getType());
    if (inReShapeInputType.getShape() != outReShapeOutputType.getShape()) {
        _log.trace("[{0}] SoftmaxOp '{1}' has different reshape input and output shapes", getDebugName(),
                   softMaxOp->getName());
        return mlir::failure();
    }

    // 4. check that inReshapeOp reshapes from [1,N*M,X,Y] to [N,M,X,Y]
    const auto inReShapeInputShape = inReShapeInputType.getShape().toValues();
    const auto inReShapeOutputShape =
            mlir::cast<vpux::NDTypeInterface>(inReShapeOp.getOutput().getType()).getShape().toValues();
    if (inReShapeInputShape.size() != 4 || inReShapeOutputShape.size() != 4) {
        _log.trace("[{0}] SoftmaxOp '{1}' has unsupported reshape input or output shape", getDebugName(),
                   softMaxOp->getName());
        return mlir::failure();
    }
    if (inReShapeInputShape[Dims4D::Act::N] != 1) {
        _log.trace("[{0}] SoftmaxOp '{1}' has unsupported reshape input shape N dimension: {2}", getDebugName(),
                   softMaxOp->getName(), inReShapeInputShape[Dims4D::Act::N]);
        return mlir::failure();
    }
    if (inReShapeInputShape[Dims4D::Act::C] !=
        inReShapeOutputShape[Dims4D::Act::N] * inReShapeOutputShape[Dims4D::Act::C]) {
        _log.trace("[{0}] SoftmaxOp '{1}' has unsupported reshape input shape C dimension: {2} != {3} * {4}",
                   getDebugName(), softMaxOp->getName(), inReShapeInputShape[Dims4D::Act::C],
                   inReShapeOutputShape[Dims4D::Act::N], inReShapeOutputShape[Dims4D::Act::C]);
        return mlir::failure();
    }

    // 5. remove inReShapeOp and outReShapeOp and use their input and output as softmax input and output
    rewriter.setInsertionPointAfter(inReShapeOp);
    auto newSoftMaxOp = rewriter.replaceOpWithNewOp<IE::SoftMaxOp>(inReShapeOp, inReShapeOp.getInput(),
                                                                   softMaxOp.getAxisInd(), softMaxOp.getPadSizeAttr());
    outReShapeOp.replaceAllUsesWith(newSoftMaxOp.getOperation());

    return mlir::success();
}

// ExpandSoftmaxAxisPass
//

class ExpandSoftmaxAxisPass final : public IE::impl::ExpandSoftmaxAxisBase<ExpandSoftmaxAxisPass> {
public:
    explicit ExpandSoftmaxAxisPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ExpandSoftmaxAxisPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SimplifyReshapes>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
    func->walk([&](IE::ExpandOp expandOp) {
        auto smOp = expandOp.getOperand().getDefiningOp<IE::SoftMaxOp>();
        if (!smOp) {
            return;
        }
        if (!smOp.getOutput().hasOneUse() && !smOp.getInput().hasOneUse()) {
            return;
        }

        auto sliceOp = smOp->getOperand(0).getDefiningOp<IE::SliceOp>();
        if (!sliceOp) {
            return;
        }

        const auto inType = mlir::cast<vpux::NDTypeInterface>(smOp->getOperand(0).getType());
        const auto inShape = inType.getShape().toValues();
        const long int inRank = checked_cast<size_t>(inType.getRank());
        const long int axis = smOp.getAxisInd();
        const auto axisDim = inShape[Dim(axis)];
        if (axis != inRank - 1) {
            return;
        }

        if (!expandOp.getPadsBegin() || !expandOp.getPadsEnd()) {
            return;
        }
        llvm::SmallVector<int64_t> padsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
        llvm::SmallVector<int64_t> padsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
        if (!arePadsValid(padsBegin, padsEnd)) {
            return;
        }
        int64_t padValue = padsEnd[axis] + (smOp.getPadSize().has_value() ? smOp.getPadSize().value() : 0);
        if (axisDim % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0 &&
            ((axisDim + padValue) % ALIGNMENT_REQUIREMENT_IN_ELEMENTS) != 0) {
            return;
        }
        auto builder = mlir::OpBuilder(smOp);
        auto newSMOp = builder.create<IE::SoftMaxOp>(appendLoc(smOp->getLoc(), "_padded"), sliceOp->getOperand(0),
                                                     smOp.getAxisIndAttr(), getIntAttr(builder.getContext(), padValue));
        expandOp->replaceAllUsesWith(newSMOp);
    });
}

}  // namespace

//
// createExpandSoftmaxAxisPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createExpandSoftmaxAxisPass(Logger log) {
    return std::make_unique<ExpandSoftmaxAxisPass>(log);
}
