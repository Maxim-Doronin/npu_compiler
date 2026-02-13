//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEREDUCEMEANSQUARE
#define GEN_PASS_DEF_FUSEREDUCEMEANSQUARE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseReduceMeanSquarePass
//

class FuseReduceMeanSquarePass final : public IE::impl::FuseReduceMeanSquareBase<FuseReduceMeanSquarePass> {
public:
    explicit FuseReduceMeanSquarePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::Operation* getPowerOp(mlir::Operation* op) {
    // detect Pow(x,2) or Multiply(x,x)
    if (auto power = mlir::dyn_cast<IE::PowerOp>(op)) {
        auto constOp = power.getInput2().getDefiningOp<Const::DeclareOp>();
        if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
            return nullptr;
        }
        const auto coefContent = constOp.getContent();
        const auto coefValue = coefContent.getSplatValue<double>();
        return (coefValue == 2.0) ? op : nullptr;
    } else if (auto mul = mlir::dyn_cast<IE::MultiplyOp>(op)) {
        return mul.getInput1() == mul.getInput2() ? op : nullptr;
    }
    return nullptr;
}

std::optional<mlir::FloatAttr> extractEpsilon(IE::AddOp addOp, mlir::OpBuilder& builder, Logger log) {
    auto maybeConstant = addOp->getOperand(0).getDefiningOp();
    auto epsilonConstOp = mlir::dyn_cast<Const::DeclareOp>(mlir::isa_and_nonnull<Const::DeclareOp>(maybeConstant)
                                                                   ? maybeConstant
                                                                   : addOp->getOperand(1).getDefiningOp());

    if (epsilonConstOp == nullptr) {
        log.trace("Epsilon operand is not a Const::DeclareOp, skipping fuse");
        return std::nullopt;
    }

    auto epsilonValue = Const::getSplatValue<float>(epsilonConstOp);
    if (mlir::failed(epsilonValue)) {
        log.trace("Failed to extract epsilon splat value, skipping fuse");
        return std::nullopt;
    }

    return getFPAttr(builder.getContext(), epsilonValue.value());
}

IE::ReduceMeanSquareOp createReduceMeanSquareOp(mlir::OpBuilder& builder, mlir::Value input, mlir::FloatAttr epsilon,
                                                mlir::ArrayAttr axesAttr, mlir::UnitAttr keepDimsAttr) {
    auto loc = input.getLoc();
    return builder.create<IE::ReduceMeanSquareOp>(appendLoc(loc, "_reduce_mean_square"), input, nullptr, epsilon,
                                                  axesAttr, keepDimsAttr);
}

void FuseReduceMeanSquarePass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](mlir::Operation* op) {
        auto powerOp = getPowerOp(op);
        if (powerOp == nullptr || !powerOp->hasOneUse()) {
            _log.trace("Power op not found or has multiple uses, skipping fuse.");
            return;
        }
        auto reduceMeanOp = mlir::dyn_cast<IE::ReduceMeanOp>(*powerOp->getUsers().begin());
        if (reduceMeanOp == nullptr || !reduceMeanOp->hasOneUse()) {
            _log.trace("ReduceMean op not found or has multiple uses, skipping fuse.");
            return;
        }

        auto userOp = *reduceMeanOp->getUsers().begin();
        auto builder = mlir::OpBuilder(reduceMeanOp);
        mlir::FloatAttr epsilonAttr = nullptr;

        if (!reduceMeanOp.getAxesValue().has_value()) {
            _log.trace("Axes value not available, skipping fuse.");
            return;
        }
        auto axesAttr = reduceMeanOp.getAxesValueAttr();

        mlir::Operation* sqrtOp = userOp;

        if (auto addOp = mlir::dyn_cast<IE::AddOp>(userOp)) {
            if (!addOp->hasOneUse()) {
                _log.trace("Add op has multiple uses, skipping fuse.");
                return;
            }

            auto extractedEpsilon = extractEpsilon(addOp, builder, _log);
            if (!extractedEpsilon.has_value()) {
                _log.trace("Failed to extract epsilon, skipping fuse.");
                return;
            }
            epsilonAttr = extractedEpsilon.value();
            const auto axes = parseIntArrayAttr<int64_t>(axesAttr);
            const auto inputRank = mlir::cast<vpux::NDTypeInterface>(reduceMeanOp.getInput().getType()).getRank();

            if (axes.size() != 1 || axes[0] != inputRank - 1) {
                _log.trace("Axis is not innermost. Skipping fuse.");
                return;
            }

            sqrtOp = *addOp->getUsers().begin();
        }

        auto sqrtOpCasted = mlir::dyn_cast<IE::SqrtOp>(sqrtOp);
        if (sqrtOpCasted == nullptr) {
            _log.trace("Sqrt op not found, skipping fuse.");
            return;
        }

        auto keepDimsAttr = reduceMeanOp.getKeepDimsAttr();

        auto reduceMeanSquareOp =
                createReduceMeanSquareOp(builder, powerOp->getOperand(0), epsilonAttr, axesAttr, keepDimsAttr);

        sqrtOpCasted->replaceAllUsesWith(reduceMeanSquareOp);
        _log.trace("[FuseReduceMeanSquare] Pattern matched");
    });
}

}  // namespace

//
// createFuseReduceMeanSquarePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseReduceMeanSquarePass(Logger log) {
    return std::make_unique<FuseReduceMeanSquarePass>(log);
}
