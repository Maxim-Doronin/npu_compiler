//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::NegativeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NegativeOpAdaptor negative(operands, attrs, prop);
    if (mlir::failed(negative.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(negative.getInput().getType());

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// FoldNegative
//

namespace {

class FoldNegative final : public mlir::OpRewritePattern<IE::NegativeOp> {
public:
    using mlir::OpRewritePattern<IE::NegativeOp>::OpRewritePattern;

private:
    mlir::LogicalResult matchAndRewrite(IE::NegativeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FoldNegative::matchAndRewrite(IE::NegativeOp origOp, mlir::PatternRewriter& /*rewriter*/) const {
    auto prevOp = origOp.getInput().getDefiningOp<IE::NegativeOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    origOp.replaceAllUsesWith(prevOp.getInput());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::NegativeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FoldNegative>(ctx);
}
