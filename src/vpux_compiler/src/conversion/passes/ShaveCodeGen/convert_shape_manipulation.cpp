//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/conversions.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

static mlir::LogicalResult convertViewLikeOp(mlir::Operation* op, mlir::Value input, mlir::PatternRewriter& rewriter) {
    // Collapse to a 1D tensor then expand to the output memory shape.
    // This sequence should then be canonicalized if possible.
    // At the moment we don't do anything more complex since there is no evidence
    // that it would justify the added complexity.
    auto loc = op->getLoc();

    auto numElem = mlir::cast<NDTypeInterface>(input.getType()).getNumElements();
    auto elTy = mlir::cast<NDTypeInterface>(input.getType()).getElementType();
    auto inputRank = mlir::cast<NDTypeInterface>(input.getType()).getRank();

    // Collapse to a 1d tensor
    SmallVector<int64_t> collapseResultShape(1, numElem);
    mlir::ReassociationIndices dimCollapse(inputRank);
    std::iota(std::begin(dimCollapse), std::end(dimCollapse), 0);
    SmallVector<mlir::ReassociationIndices> collapseReassocMap;
    collapseReassocMap.emplace_back(dimCollapse);
    auto collapseResultType = mlir::RankedTensorType::get(collapseResultShape, elTy);

    input = rewriter.create<mlir::tensor::CollapseShapeOp>(loc, collapseResultType, input, collapseReassocMap);

    // Expand to target shape
    auto ndResultTy = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto resultDimsOrder = DimsOrder::fromPermutation(ndResultTy.getDimsOrder().toPermutation());
    auto resultShape = resultDimsOrder.toMemoryOrder(ndResultTy.getShape()).raw();
    auto expandResultType = mlir::RankedTensorType::get(resultShape, elTy);

    mlir::ReassociationIndices dimExpand(ndResultTy.getRank());
    std::iota(std::begin(dimExpand), std::end(dimExpand), 0);
    SmallVector<mlir::ReassociationIndices> expandReassocMap;
    expandReassocMap.emplace_back(dimExpand);

    input = rewriter.create<mlir::tensor::ExpandShapeOp>(loc, expandResultType, input, expandReassocMap);
    rewriter.replaceOp(op, input);

    return mlir::success();
}

template <typename SrcOp>
class IEViewLikeToCollapseExpand : public mlir::OpConversionPattern<SrcOp> {
public:
    using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;
    using OpAdaptor = typename mlir::OpConversionPattern<SrcOp>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto rankedOutTy = mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
        if (adaptor.getOperands()[0].getType() == ShaveCodeGen::normalizeType(rankedOutTy)) {
            rewriter.replaceOp(op, adaptor.getOperands()[0]);
            return mlir::success();
        }
        return convertViewLikeOp(op, adaptor.getOperands()[0], rewriter);
    }
};

}  // namespace

void ShaveCodeGen::populateIEShapeManipulationToTensorPatterns(mlir::RewritePatternSet& patternSet,
                                                               mlir::TypeConverter& typeConverter) {
    auto& ctx = *patternSet.getContext();
    patternSet.add<IEViewLikeToCollapseExpand<IE::PermuteCastOp>, IEViewLikeToCollapseExpand<IE::AffineReshapeOp>>(
            typeConverter, &ctx);
}
