//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSESOFTMAXCONVERT
#define GEN_PASS_DEF_FUSESOFTMAXCONVERT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseSoftMaxConvert
//

class FuseSoftMaxConvertPass final : public IE::impl::FuseSoftMaxConvertBase<FuseSoftMaxConvertPass> {
public:
    explicit FuseSoftMaxConvertPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

bool isLegalSoftMax(IE::SoftMaxOp softMaxOp) {
    if (!softMaxOp->hasOneUse()) {
        return false;
    }

    int64_t softMaxAxis = parseIntAttr<int64_t>(softMaxOp.getAxisIndAttr());

    const auto inOrder = DimsOrder::fromValue(softMaxOp.getInput());

    if (softMaxAxis < 0) {
        softMaxAxis += inOrder.numDims();
    }
    MemDim md = inOrder.toMemDim(Dim(softMaxAxis));

    const auto shape = getShape(softMaxOp.getInput());
    auto nDims = checked_cast<uint32_t>(shape.size());

    // Currently only the inner mode is supported
    if (md.ind() != (int32_t)(nDims - 1)) {
        return false;
    }

    return true;
}

mlir::FailureOr<IE::ConvertOp> findUserConvert(IE::SoftMaxOp softMaxOp) {
    mlir::Operation* userOp = softMaxOp.getOperation();

    while (userOp->hasOneUse()) {
        userOp = *userOp->getUsers().begin();

        if (auto convertOp = mlir::dyn_cast<IE::ConvertOp>(userOp)) {
            auto convertOutputType = mlir::cast<vpux::NDTypeInterface>(convertOp.getOutput().getType());
            if (convertOutputType.getElementType().isF32()) {
                return convertOp;
            }
            return mlir::failure();
        }

        if (!IE::isPureViewOp(userOp) && !mlir::isa<IE::SliceOp>(userOp)) {
            return mlir::failure();
        }
    }

    return mlir::failure();
}

void updatePrecisions(IE::SoftMaxOp softMaxOp, mlir::Type newElemType) {
    mlir::Operation* userOp = softMaxOp.getOperation();

    while (userOp->hasOneUse()) {
        userOp = *userOp->getUsers().begin();

        if (mlir::isa<IE::ConvertOp>(userOp)) {
            break;
        }

        assert((IE::isPureViewOp(userOp) || mlir::isa<IE::SliceOp>(userOp)) && "Unexpected intermediate operation");

        // The findUserConvert function ensures these are the expected operations
        auto currentType = mlir::cast<vpux::NDTypeInterface>(userOp->getResult(0).getType());
        auto newType = currentType.changeElemType(newElemType);
        userOp->getResult(0).setType(newType);
    }

    auto currentType = mlir::cast<vpux::NDTypeInterface>(softMaxOp->getResult(0).getType());
    auto newType = currentType.changeElemType(newElemType);
    softMaxOp->getResult(0).setType(newType);
    softMaxOp.setDstElemType(newElemType);
}

class FuseSoftMaxConvertPattern final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    FuseSoftMaxConvertPattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        setDebugName("FuseSoftMaxConvertPattern");
    }

    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp softMaxOp, mlir::PatternRewriter& rewriter) const final {
        if (!isLegalSoftMax(softMaxOp)) {
            return mlir::failure();
        }

        auto softMaxOutputType = mlir::cast<vpux::NDTypeInterface>(softMaxOp.getOutput().getType());
        if (!softMaxOutputType.getElementType().isF16()) {
            return mlir::failure();
        }

        auto convertOpResult = findUserConvert(softMaxOp);
        if (mlir::failed(convertOpResult)) {
            return mlir::failure();
        }

        auto convertOp = *convertOpResult;

        _log.trace("SoftMax -> ... -> ConvertFP32 pattern matched for operation {0} at {1}", convertOp->getName(),
                   convertOp->getLoc());

        // Update all operation types to F32
        auto f32Type = mlir::Float32Type::get(softMaxOp->getContext());
        updatePrecisions(softMaxOp, f32Type);

        rewriter.replaceAllUsesWith(convertOp, convertOp.getInput());

        return mlir::success();
    }

private:
    Logger _log;
};

//
// safeRunOnFunc
//

void FuseSoftMaxConvertPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseSoftMaxConvertPattern>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseSoftMaxConvertPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseSoftMaxConvertPass(Logger log) {
    return std::make_unique<FuseSoftMaxConvertPass>(log);
}
