//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGATHER
#define GEN_PASS_DEF_CONVERTGATHER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertGatherPass
//

class ConvertGatherPass final : public IE::impl::ConvertGatherBase<ConvertGatherPass> {
public:
    explicit ConvertGatherPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GatherToSlice;
    class GatherToReverse;

private:
    void safeRunOnFunc() final;
};

bool checkAttrsForGatherOp(IE::GatherOp gatherOp) {
    const auto batchDims = gatherOp.getBatchDims();

    auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    if (indices == nullptr) {
        return false;
    }

    return batchDims == 0 && gatherOp.getAxisValue().has_value();
}

//
// GatherToSlice
//

class ConvertGatherPass::GatherToSlice final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    GatherToSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("ConvertGatherPass::GatherToSlice");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGatherPass::GatherToSlice::matchAndRewrite(IE::GatherOp gatherOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Gather Op: {0}", gatherOp);
    auto* ctx = rewriter.getContext();

    if (!checkAttrsForGatherOp(gatherOp)) {
        return mlir::failure();
    }

    auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    const auto indicesContent = indices.getContent();
    if (indicesContent.getType().getNumElements() != 1) {
        return mlir::failure();
    }

    const auto indicesVal = indicesContent.getSplatValue<int64_t>();

    const auto axisVal = gatherOp.getAxisValue().value();

    const auto inType = mlir::cast<vpux::NDTypeInterface>(gatherOp.getInput().getType());
    const auto inputShape = inType.getShape();
    auto staticOffsets = SmallVector<int64_t>(inputShape.size(), 0);
    staticOffsets[axisVal] = indicesVal;

    SmallVector<int64_t> staticSizes(inputShape.begin(), inputShape.end());
    staticSizes[axisVal] = 1;

    const auto sliceOpLoc = appendLoc(gatherOp.getLoc(), "slice");
    auto sliceOp = rewriter.create<IE::SliceOp>(sliceOpLoc, gatherOp.getInput(), getIntArrayAttr(ctx, staticOffsets),
                                                getIntArrayAttr(ctx, staticSizes));

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(gatherOp, sliceOp.getResult(),
                                               getIntArrayAttr(ctx, getShape(gatherOp.getOutput())));

    return mlir::success();
}

//
// GatherToReverse
//

class ConvertGatherPass::GatherToReverse final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    GatherToReverse(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("ConvertGatherPass::GatherToReverse");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGatherPass::GatherToReverse::matchAndRewrite(IE::GatherOp gatherOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), gatherOp->getName(), gatherOp->getLoc());
    if (!checkAttrsForGatherOp(gatherOp)) {
        return mlir::failure();
    }

    auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    const auto indicesContent = indices.getContent();
    const auto indicesNums = indicesContent.getType().getNumElements();

    if (indicesNums == 1) {
        _log.trace("[{0}] Only one index value, converting to Slice", this->getDebugName());
        return mlir::failure();
    }

    const auto axisVal = gatherOp.getAxisValue().value();
    const auto inputShape = getShape(gatherOp.getInput());

    // For GatherDMA all dimensions before axis dimension must be 1
    if (std::all_of(inputShape.begin(), inputShape.begin() + axisVal, [](int64_t dim) {
            return dim == 1;
        })) {
        _log.trace("[{0}] All dimensions before axis dimension are 1, converting to GatherDMA", this->getDebugName());
        return mlir::failure();
    }

    if (inputShape[Dim(axisVal)] != indicesNums || inputShape != getShape(gatherOp.getOutput())) {
        _log.trace("[{0}] Input shape does not match indices number or output shape, cannot convert to Reverse",
                   this->getDebugName());
        return mlir::failure();
    }

    auto areIndicesReverseContiguous = [](const SmallVector<int64_t>& indicesValues) -> bool {
        auto it = std::adjacent_find(indicesValues.begin(), indicesValues.end(), [](int64_t prev, int64_t curr) {
            return curr != prev - 1;
        });
        return it == indicesValues.end();
    };

    const auto vals = to_small_vector(indicesContent.getValues<int64_t>());
    if (vals.empty() || !areIndicesReverseContiguous(vals)) {
        _log.trace("[{0}] Indices are not reverse contiguous, cannot convert to Reverse", this->getDebugName());
        return mlir::failure();
    }

    const auto ctx = gatherOp.getContext();
    const auto axisAttr = getIntArrayAttr(ctx, ArrayRef(axisVal));
    const auto modeAttr = IE::ReverseModeAttr::get(ctx, IE::ReverseMode::INDEX);
    auto reverseOp =
            rewriter.create<IE::ReverseOp>(gatherOp.getLoc(), gatherOp.getInput(), nullptr, axisAttr, modeAttr);

    _log.trace("[{0}] Replaced with Reverse '{1}' at '{2}'", this->getDebugName(), gatherOp->getName(),
               gatherOp->getLoc());
    rewriter.replaceOp(gatherOp, reverseOp.getResult());
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertGatherPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GatherToSlice>(&ctx, _log);
    patterns.add<GatherToReverse>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGatherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGatherPass(Logger log) {
    return std::make_unique<ConvertGatherPass>(log);
}
