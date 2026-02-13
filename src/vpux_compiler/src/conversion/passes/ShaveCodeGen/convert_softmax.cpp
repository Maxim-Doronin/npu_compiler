//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/conversions.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

class IESoftMaxToLinalg : public mlir::OpConversionPattern<IE::SoftMaxOp> {
public:
    using mlir::OpConversionPattern<IE::SoftMaxOp>::OpConversionPattern;
    using OpAdaptor = mlir::OpConversionPattern<IE::SoftMaxOp>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult IESoftMaxToLinalg::matchAndRewrite(IE::SoftMaxOp op, OpAdaptor adaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    auto inputOrder = DimsOrder::fromValue(op->getOperand(0)).toAffineMap(rewriter.getContext());
    auto outputOrder = DimsOrder::fromValue(op->getResult(0)).toAffineMap(rewriter.getContext());
    auto input = adaptor.getOperands()[0];
    // The softmax axis in input memory order.
    auto axis = mlir::cast<mlir::AffineDimExpr>(mlir::inversePermutation(inputOrder).getResult(op.getAxisInd()))
                        .getPosition();
    // The softmax axis in the output memory order.
    auto outAxis = mlir::cast<mlir::AffineDimExpr>(mlir::inversePermutation(outputOrder).getResult(op.getAxisInd()))
                           .getPosition();
    auto resultType = ShaveCodeGen::normalizeType(mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()));

    auto elTy = ShaveCodeGen::getLinalgElementType(op->getOperand(0).getType(), rewriter.getContext());
    auto rank = mlir::cast<vpux::NDTypeInterface>(input.getType()).getRank();
    auto loc = op->getLoc();
    bool changesOrder = inputOrder != outputOrder;

    auto pad = op.getPadSize();
    bool hasPad = pad && *pad != 0;

    // Output tensor of the linalg.softmax op
    mlir::Value out = nullptr;
    // Output padded tensor
    mlir::Value padded = nullptr;
    // Output tensor of the linalg.transpose or linalg.softmax (if linalg.transpose is not needed).
    mlir::Value sliceOut = nullptr;

    if (hasPad) {
        // Emit the extract slice on the input to remove padding.
        auto shapeRef = mlir::cast<mlir::ShapedType>(input.getType()).getShape();
        SmallVector<int64_t> shape(shapeRef.begin(), shapeRef.end());
        shape[axis] -= *pad;

        SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
        SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
        SmallVector<mlir::OpFoldResult> sizes(rank);
        for (unsigned i = 0; i < rank; ++i) {
            sizes[i] = rewriter.getIndexAttr(shape[i]);
        }

        input = rewriter.create<mlir::tensor::ExtractSliceOp>(loc, input, offsets, sizes, strides);

        // Create the padded output tensor (initialized with zeros) and an
        // extract_slice to act as the unpadded output.
        auto outShapeRef = mlir::cast<mlir::ShapedType>(resultType).getShape();
        SmallVector<int64_t> outShape(outShapeRef.begin(), outShapeRef.end());
        outShape[outAxis] -= *pad;

        std::tie(sliceOut, padded) = ShaveCodeGen::emitTensorSlice(loc, outShape, resultType, rewriter);
        if (!changesOrder) {
            // The output order is the same as the input order so we're able to write directly
            // to the output slice.
            out = sliceOut;
        }
    }

    if (out == nullptr) {
        // No output tensor for the softmax op has yet been created, so either we're not using padding or we need
        // to emit a transpose op. In either case we need create the output tensor as a tensor.empty.
        out = rewriter.create<mlir::tensor::EmptyOp>(loc, mlir::cast<mlir::ShapedType>(input.getType()).getShape(),
                                                     elTy);
    }

    auto linalgOp = rewriter.create<mlir::linalg::SoftmaxOp>(loc, mlir::TypeRange{input.getType()}, input, out, axis);

    mlir::Value result = linalgOp->getResult(0);

    if (changesOrder) {
        if (sliceOut == nullptr) {
            // Create the output tensor for the transpose op if we don't have one yet (this can happen
            // when no padding is applied).
            sliceOut = rewriter.create<mlir::tensor::EmptyOp>(loc, mlir::cast<mlir::ShapedType>(resultType).getShape(),
                                                              elTy);
        }

        // Permute the result of the softmax op to match the output order.
        SmallVector<int64_t> perm(rank);
        auto transposePerm = outputOrder.compose(mlir::inversePermutation(inputOrder));
        for (int i = 0; i < rank; ++i) {
            perm[i] = mlir::cast<mlir::AffineDimExpr>(transposePerm.getResult(i)).getPosition();
        }
        result = rewriter.create<mlir::linalg::TransposeOp>(loc, result, sliceOut, perm)->getResult(0);
    }

    if (hasPad) {
        // Insert the softmax result in the previously generated padding.
        SmallVector<mlir::OpFoldResult> zeros(rank, rewriter.getIndexAttr(0));
        SmallVector<mlir::OpFoldResult> ones(rank, rewriter.getIndexAttr(1));
        SmallVector<mlir::OpFoldResult> sizes = mlir::tensor::getMixedSizes(rewriter, loc, result);
        result = rewriter.create<mlir::tensor::InsertSliceOp>(loc, result, padded, zeros, sizes, ones)->getResult(0);
    }

    rewriter.replaceOp(op, result);

    return mlir::success();
}

}  // namespace

void ShaveCodeGen::populateIESoftmaxToLinalgPatterns(mlir::RewritePatternSet& patternSet,
                                                     mlir::TypeConverter& typeConverter) {
    patternSet.add<IESoftMaxToLinalg>(typeConverter, patternSet.getContext());
}
