//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Tensor/Transforms/Transforms.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_FOLDUNITDIMRESHAPES
#define GEN_PASS_DEF_FOLDUNITDIMRESHAPES
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// FoldUnitDimReshapesPass
//

class FoldUnitDimReshapesPass final : public ShaveCodeGen::impl::FoldUnitDimReshapesBase<FoldUnitDimReshapesPass> {
public:
    explicit FoldUnitDimReshapesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnFunc() final;
    mlir::LogicalResult runOnCapsule(IE::CodeGenCapsuleOp op, const mlir::FrozenRewritePatternSet& foldUnitDimPatterns,
                                     const mlir::FrozenRewritePatternSet& reshapePatterns);
};

mlir::LogicalResult FoldUnitDimReshapesPass::runOnCapsule(IE::CodeGenCapsuleOp op,
                                                          const mlir::FrozenRewritePatternSet& foldUnitDimPatterns,
                                                          const mlir::FrozenRewritePatternSet& reshapePatterns) {
    // Remove unit dimensions to align compute shape.
    if (failed(mlir::applyPatternsGreedily(op, foldUnitDimPatterns, getDefaultGreedyRewriteConfig()))) {
        return mlir::failure();
    }

    // Apply the reshape patterns bottom-up since we want to push the reshapes at the beginning
    // of the capsule. Note that these will also interfere with the fold unit dims patterns
    // and need to run in as a separate traversal.
    mlir::GreedyRewriteConfig grc;
    grc.setUseTopDownTraversal(false);
    if (failed(mlir::applyPatternsGreedily(op, reshapePatterns, grc))) {
        return mlir::failure();
    }

    return mlir::success();
}

void FoldUnitDimReshapesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto funcOp = getOperation();

    // Run two phases of optimizations to remove unit dimension reshape chains.
    // First, squeeze tensor shapes to align the shape of compute operations and apply
    // reshape optimizations.
    mlir::RewritePatternSet foldUnitDimsPatterns(&ctx);
    mlir::linalg::ControlDropUnitDims options;
    options.controlFn = [](mlir::Operation* op) {
        if (auto genericOp = mlir::dyn_cast_or_null<mlir::linalg::GenericOp>(op)) {
            bool hasScalarInputs = llvm::all_of(op->getOperands(), [](mlir::Value val) {
                auto ty = mlir::dyn_cast<mlir::ShapedType>(val.getType());
                if (!ty || !ty.hasRank()) {
                    return false;
                }
                return llvm::all_of(ty.getShape(), [](int64_t dim) {
                    return dim == 1;
                });
            });
            if (hasScalarInputs) {
                // If all input/output shapes are all ones keep the last dimension so
                // that reshapes can still be propagated upwards through this op.
                // This works around a limitation of FoldReshapeOpsByExpansion.
                return llvm::to_vector(llvm::seq<unsigned>(1, genericOp.getNumLoops()));
            }

            return llvm::to_vector(llvm::seq<unsigned>(0, genericOp.getNumLoops()));
        }
        return SmallVector<unsigned>{};
    };
    mlir::linalg::populateFoldUnitExtentDimsPatterns(foldUnitDimsPatterns, options);
    const mlir::FrozenRewritePatternSet frozenFoldUnitDimsPatterns(std::move(foldUnitDimsPatterns));

    // Second, push the remaining reshapes up as much as possible through linalg ops.
    // This helps with removing reshapes between ops, enabling fusion (both at linalg level and scf),
    // and will help with empty tensor elimination as well.
    mlir::RewritePatternSet reshapePatterns(&ctx);
    mlir::linalg::populateFoldReshapeOpsByExpansionPatterns(reshapePatterns, [](mlir::OpOperand* fusedOperand) {
        auto op = fusedOperand->get().getDefiningOp();
        if (!mlir::isa<mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(op)) {
            return true;
        }
        return false;
    });
    mlir::tensor::populateFoldTensorEmptyPatterns(reshapePatterns);
    mlir::tensor::CollapseShapeOp::getCanonicalizationPatterns(reshapePatterns, &ctx);
    mlir::tensor::EmptyOp::getCanonicalizationPatterns(reshapePatterns, &ctx);
    mlir::tensor::ExpandShapeOp::getCanonicalizationPatterns(reshapePatterns, &ctx);
    const mlir::FrozenRewritePatternSet frozenReshapePatterns(std::move(reshapePatterns));

    funcOp->walk([&](IE::CodeGenCapsuleOp capsuleOp) {
        if (mlir::failed(runOnCapsule(capsuleOp, frozenFoldUnitDimsPatterns, frozenReshapePatterns))) {
            return signalPassFailure();
        }
    });

    return;
}

}  // namespace

//
// createFoldUnitDimReshapesPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createFoldUnitDimReshapesPass(Logger log) {
    return std::make_unique<FoldUnitDimReshapesPass>(log);
}
