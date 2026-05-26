//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEPERMUTECAST
#define GEN_PASS_DEF_PROPAGATEPERMUTECAST
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// PermuteCastRewriter
//

class PropagateThroughDequantize final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    PropagateThroughDequantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteCastOp>(ctx), _log(log) {
        this->setDebugName("PropagateThroughDequantize");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isSupportedDequantizeLayout(IE::PermuteCastOp permuteCastOp, IE::DequantizeOp dequantizeOp) {
    // Doing similar check to verifyDequantizeLayoutInfo with same rules
    const auto permutedCastOutput = permuteCastOp.getOutput();
    const auto permuteCastOutputOrder = DimsOrder::fromValue(permutedCastOutput);
    const auto inputType = mlir::cast<NDTypeInterface>(dequantizeOp.getInput().getType());
    const auto quantizedType = mlir::cast<mlir::quant::QuantizedType>(inputType.getElementType());
    SmallVector<DimsOrder> supportedLayouts;
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(quantizedType)) {
        const auto numDims = inputType.getRank();
        if (numDims == 3) {
            supportedLayouts = {DimsOrder::HWC};
        } else if (numDims == 4) {
            supportedLayouts = {DimsOrder::NHWC};
        } else {
            VPUX_THROW("Unsupported rank '{0}'", numDims);
        }
    } else {
        supportedLayouts = {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC};
    }
    return std::find(supportedLayouts.begin(), supportedLayouts.end(), permuteCastOutputOrder) !=
           supportedLayouts.end();
}

mlir::LogicalResult PropagateThroughDequantize::matchAndRewrite(IE::PermuteCastOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto parentDequantOp = origOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (!parentDequantOp) {
        _log.nest().trace("Input is not produced by DequantizeOp");
        return mlir::failure();
    }

    if (!isSupportedDequantizeLayout(origOp, parentDequantOp)) {
        _log.nest().trace("Unsupported Dequantize layout for propagation");
        return mlir::failure();
    }
    auto newPermuteCastOp = rewriter.createOrFold<IE::PermuteCastOp>(origOp.getLoc(), parentDequantOp.getInput(),
                                                                     origOp.getDstOrder(), origOp.getMemPerm());
    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(origOp, newPermuteCastOp, parentDequantOp.getDstElemType());
    _log.nest().trace("Propagated PermuteCast through DequantizeOp at '{0}'", parentDequantOp->getLoc());
    return mlir::success();
}

SmallVector<int64_t> remapDimsThroughPermuteCast(ArrayRef<int64_t> dims, NDTypeInterface inType,
                                                 NDTypeInterface outType, mlir::AffineMap memPerm) {
    const auto inMemOrder = inType.getDimsOrder().toMemoryOrder(Shape(dims));
    const auto permuted = applyPerm(inMemOrder, memPerm);
    return outType.getDimsOrder().toLogicalOrder(permuted).raw();
}

//
// PropagateThroughConcat
//

class PropagateThroughConcat final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    PropagateThroughConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteCastOp>(ctx), _log(log) {
        this->setDebugName("PropagateThroughConcat");
    }

    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
        auto concatOp = origOp.getInput().getDefiningOp<IE::ConcatOp>();
        if (mlir::failed(matchPattern(origOp))) {
            _log.nest().trace("Pattern match failed for PropagateThroughConcat");
            return mlir::failure();
        }

        mlir::SmallVector<mlir::Value> newConcatInputs;
        for (auto [index, input] : concatOp.getInputs() | indexed) {
            auto newPermuteCast = rewriter.createOrFold<IE::PermuteCastOp>(
                    appendLoc(origOp->getLoc(), "input_{}", index), input, origOp.getDstOrder(), origOp.getMemPerm());
            newConcatInputs.push_back(newPermuteCast);
        }

        const auto staticOffsets = getStaticOffsets(concatOp);
        const auto inType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
        const auto outType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
        auto staticOffsetsAttr = parseIntArrayOfArrayAttr<int64_t>(staticOffsets.value());
        for (auto& offsets : staticOffsetsAttr) {
            offsets = remapDimsThroughPermuteCast(offsets, inType, outType, origOp.getMemPerm());
        }

        auto newConcat = rewriter.create<IE::ConcatOp>(concatOp.getLoc(), newConcatInputs, nullptr,
                                                       getIntArrayOfArray(rewriter.getContext(), staticOffsetsAttr));
        rewriter.replaceOp(origOp, newConcat.getResult());
        _log.nest().trace("Propagated PermuteCast through ConcatOp at '{0}'", concatOp->getLoc());
        return mlir::success();
    }

private:
    mlir::LogicalResult matchPattern(IE::PermuteCastOp origOp) const {
        auto concatOp = origOp.getInput().getDefiningOp<IE::ConcatOp>();
        if (!concatOp) {
            return mlir::failure();
        }
        // Only propagate through Concat with <=2 inputs and at least one from PermuteCast,
        // so the transformation does not increase the number of PermuteCast operations
        if (concatOp.getInputs().size() > 2) {
            return mlir::failure();
        }
        const auto numPermuteCastInputs = llvm::count_if(concatOp.getInputs(), [](mlir::Value input) {
            return mlir::isa_and_present<IE::PermuteCastOp>(input.getDefiningOp());
        });
        if (numPermuteCastInputs == 0) {
            return mlir::failure();
        }
        if (!getStaticOffsets(concatOp).has_value()) {
            return mlir::failure();
        }
        return mlir::success();
    }

    std::optional<mlir::ArrayAttr> getStaticOffsets(IE::ConcatOp concatOp) const {
        if (auto attr = concatOp.getStaticOffsetsAttr()) {
            return attr;
        }
        const auto perAxisAttr = concatOp.getPerAxisAttr();
        if (perAxisAttr.getStride() || perAxisAttr.getOffset()) {
            _log.nest().trace("ConcatOp has strided/offset per_axis attribute, not supported");
            return std::nullopt;
        }
        auto axis = perAxisAttr.getAxis().getValue().getSExtValue();
        return inferOffsetsAttrWithAxis(concatOp, axis);
    }

    Logger _log;
};

//
// PropagateThroughSlice
//

class PropagateThroughSlice final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    PropagateThroughSlice(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteCastOp>(ctx), _log(log) {
        this->setDebugName("PropagateThroughSlice");
    }

    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
        auto sliceOp = origOp.getInput().getDefiningOp<IE::SliceOp>();
        if (mlir::failed(matchPattern(origOp))) {
            _log.nest().trace("Pattern match failed for PropagateThroughSlice");
            return mlir::failure();
        }

        const auto inType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
        const auto outType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
        const auto memPerm = origOp.getMemPerm();

        const auto oldOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
        const auto newOffsets = remapDimsThroughPermuteCast(oldOffsets, inType, outType, memPerm);

        const auto oldSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr());
        const auto newSizes = remapDimsThroughPermuteCast(oldSizes, inType, outType, memPerm);

        auto newPermuteCast = rewriter.createOrFold<IE::PermuteCastOp>(origOp.getLoc(), sliceOp.getInput(),
                                                                       origOp.getDstOrder(), memPerm);
        auto newSlice = rewriter.create<IE::SliceOp>(sliceOp->getLoc(), newPermuteCast,
                                                     getIntArrayAttr(rewriter.getContext(), newOffsets),
                                                     getIntArrayAttr(rewriter.getContext(), newSizes));
        rewriter.replaceOp(origOp, newSlice.getResult());
        _log.nest().trace("Propagated PermuteCast through SliceOp at '{0}'", sliceOp->getLoc());
        return mlir::success();
    }

private:
    mlir::LogicalResult matchPattern(IE::PermuteCastOp origOp) const {
        auto sliceOp = origOp.getInput().getDefiningOp<IE::SliceOp>();
        if (!sliceOp) {
            return mlir::failure();
        }
        auto parentPermuteCast = sliceOp.getInput().getDefiningOp<IE::PermuteCastOp>();
        if (!parentPermuteCast) {
            return mlir::failure();
        }
        if (!areCancellingPermuteCasts(origOp, parentPermuteCast)) {
            return mlir::failure();
        }
        return mlir::success();
    }

    bool areCancellingPermuteCasts(IE::PermuteCastOp child, IE::PermuteCastOp parent) const {
        const auto composedPerm = child.getMemPerm().compose(parent.getMemPerm());
        if (!composedPerm.isIdentity()) {
            _log.nest().trace("Composed mem_perm is not identity, PermuteCasts will not cancel");
            return false;
        }
        const auto parentInputOrder = DimsOrder::fromValue(parent.getInput());
        const auto childOutputOrder = DimsOrder::fromValue(child.getOutput());
        if (parentInputOrder != childOutputOrder) {
            _log.nest().trace("Output order '{0}' does not match parent input order '{1}'", childOutputOrder,
                              parentInputOrder);
            return false;
        }
        return true;
    }

    Logger _log;
};

//
// PropagatePermuteCastPass
//

class PropagatePermuteCastPass final : public vpux::IE::impl::PropagatePermuteCastBase<PropagatePermuteCastPass> {
public:
    explicit PropagatePermuteCastPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PropagatePermuteCastPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PropagateThroughDequantize>(&ctx, _log);
    patterns.add<PropagateThroughConcat>(&ctx, _log);
    patterns.add<PropagateThroughSlice>(&ctx, _log);
    IE::PermuteCastOp::getCanonicalizationPatterns(patterns, &ctx);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagatePermuteCastPass(Logger log) {
    return std::make_unique<PropagatePermuteCastPass>(log);
}
