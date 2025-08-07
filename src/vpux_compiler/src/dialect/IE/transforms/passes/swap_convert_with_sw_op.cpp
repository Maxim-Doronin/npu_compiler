//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPCONVERTWITHSWOP
#define GEN_PASS_DEF_SWAPCONVERTWITHSWOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// TODO: Adding limitation, noting that small tensors are not that optimal to fuse in NCE
constexpr int64_t EXPERIMENTAL_F32_FUSION_THRESHOLD = 36000;

//
// SwapConvertWithSWOp
//

class SwapConvertWithSWOp final : public IE::impl::SwapConvertWithSWOpBase<SwapConvertWithSWOp> {
public:
    explicit SwapConvertWithSWOp(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class OpSwapConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

bool isReshapeKindOp(mlir::Operation* op) {
    return mlir::isa_and_nonnull<IE::TransposeOp, IE::ReshapeOp, IE::AffineReshapeOp>(op);
}
//
// OpSwapConverter
//

class SwapConvertWithSWOp::OpSwapConverter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    OpSwapConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapConvertWithSWOp::OpSwapConverter::matchAndRewrite(IE::ConvertOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    const auto convertInput = origOp.getInput();

    mlir::Operation* nceOp = convertInput.getDefiningOp();
    while (isReshapeKindOp(nceOp)) {
        nceOp = nceOp->getOperand(0).getDefiningOp();
    }

    rewriter.setInsertionPointAfter(nceOp);
    auto newConvert = rewriter.create<IE::ConvertOp>(nceOp->getLoc(), nceOp->getResult(0), origOp.getDstElemType());

    nceOp->getResult(0).replaceAllUsesExcept(newConvert.getOutput(),
                                             llvm::SmallPtrSet<mlir::Operation*, 1>{newConvert});

    origOp->replaceAllUsesWith(mlir::ValueRange(convertInput));
    rewriter.eraseOp(origOp);

    mlir::Operation* lastOp = *newConvert.getOutput().getUsers().begin();
    while (isReshapeKindOp(lastOp)) {
        vpux::inferReturnTypes(lastOp, vpux::InferShapedTypeMode::ALL);
        lastOp = *lastOp->getResult(0).getUsers().begin();
    }

    return mlir::success();
}

void SwapConvertWithSWOp::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();

    const auto isLegalOp = [](IE::ConvertOp op) -> bool {
        auto inputElemType = mlir::cast<NDTypeInterface>(op.getInput().getType()).getElementType();
        auto outputElemType = mlir::cast<NDTypeInterface>(op.getOutput().getType()).getElementType();

        auto outShape = getShape(op.getOutput());
        if (outShape.isDynamic() || outShape.totalSize() < EXPERIMENTAL_F32_FUSION_THRESHOLD) {
            return true;
        }

        if (!mlir::isa<mlir::Float16Type>(inputElemType) || !mlir::isa<mlir::Float32Type>(outputElemType)) {
            return true;
        }

        if (!op->hasOneUse()) {
            return true;
        }

        mlir::Operation* parentOp = op.getInput().getDefiningOp();
        if (!isReshapeKindOp(parentOp)) {
            return true;
        }
        while (isReshapeKindOp(parentOp)) {
            if (!parentOp->getResult(0).hasOneUse()) {
                return true;
            }
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }

        if (parentOp == nullptr || !parentOp->getResult(0).hasOneUse()) {
            return true;
        }

        const auto inputShape = getShape(parentOp->getOperand(0));
        // This will cause an error, because of EnsureNCEOpsSizeRequirementsPass.
        if (inputShape[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
            return true;
        }

        // SplitSEOpsPass will cause a compilation error for Interpolate.
        if (mlir::isa_and_nonnull<IE::InterpolateOp>(parentOp)) {
            return true;
        }

        return mlir::failed(VPU::NCEInvariant::isSupported(parentOp));
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvertOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapConvertWithSWOp::OpSwapConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapConvertWithSWOpPass(Logger log) {
    return std::make_unique<SwapConvertWithSWOp>(log);
}
