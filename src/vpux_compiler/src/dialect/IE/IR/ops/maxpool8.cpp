//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPool8Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MaxPool8OpAdaptor maxPool8(operands, attrs, prop);
    if (mlir::failed(maxPool8.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool8.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool8.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool8.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool8.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(maxPool8.getDilations());
    const auto roundingType = maxPool8.getRoundingType();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(maxPool8.getInput().getType());
    const auto inType = inputType.getElementType();
    const auto inShape = ShapeInfo::fromNDType(inputType);

    auto outputShape = inferMaxPool8OutputShape(inShape, windowStrides, windowDilations, dataPaddingBelow,
                                                dataPaddingAbove, windowShape, roundingType);

    inferredReturnShapes.emplace_back(outputShape.shape, inType);
    inferredReturnShapes.emplace_back(outputShape.shape, maxPool8.getIndexElementType());

    return mlir::success();
}

mlir::LogicalResult vpux::IE::MaxPool8Op::verify() {
    const auto inRank = mlir::cast<mlir::ShapedType>(getInput().getType()).getRank();
    auto axis = getAxis();

    axis = axis < 0 ? axis + inRank : axis;

    if (axis >= 0 && axis < inRank) {
        return mlir::success();
    }

    return mlir::failure();
}

//
// Canonicalizer
//
namespace {
class NormalizeAxisToPositive final : public mlir::OpRewritePattern<IE::MaxPool8Op> {
public:
    using mlir::OpRewritePattern<IE::MaxPool8Op>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPool8Op origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult NormalizeAxisToPositive::matchAndRewrite(IE::MaxPool8Op origOp, mlir::PatternRewriter&) const {
    auto axis = origOp.getAxis();
    if (axis < 0) {
        axis += origOp.getInput().getType().getRank();
        origOp.setAxis(axis);
    } else {
        return mlir::failure();
    }

    return mlir::success();
}

class RemoveIdentityMaxPool8 final : public mlir::OpRewritePattern<IE::MaxPool8Op> {
public:
    using mlir::OpRewritePattern<IE::MaxPool8Op>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPool8Op origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult RemoveIdentityMaxPool8::matchAndRewrite(IE::MaxPool8Op origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());

    bool isIdentity = llvm::all_of(strides,
                                   [](int64_t s) {
                                       return s == 1;
                                   }) &&
                      llvm::all_of(kernelSize, [](int64_t k) {
                          return k == 1;
                      });

    if (!isIdentity) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> finalResults;
    // output-0
    finalResults.push_back(origOp.getInput());

    // output-1, const
    const auto outputShape = origOp.getInput().getType().getShape();
    const auto outputType = mlir::RankedTensorType::get(outputShape, origOp.getIndexElementType());
    const auto outputSize =
            std::accumulate(outputShape.begin(), outputShape.end(), int64_t(1), std::multiplies<int64_t>());
    const auto axis = origOp.getAxis();

    int64_t modulo = outputSize;
    for (int64_t i = 0; i < outputType.getRank(); ++i) {
        if (i == axis) {
            break;
        }
        modulo /= outputType.getDimSize(i);
    }

    SmallVector<int64_t> outputValues(outputSize);
    for (int64_t i = 0; i < outputSize; i++) {
        outputValues[i] = i % modulo;
    }

    auto output =
            Const::createConst(rewriter, appendLoc(origOp.getLoc(), "_index"), outputType, ArrayRef(outputValues));
    finalResults.push_back(output);

    rewriter.replaceOp(origOp, finalResults);
    return mlir::success();
}

}  // namespace

void vpux::IE::MaxPool8Op::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<NormalizeAxisToPositive>(context);
    patterns.add<RemoveIdentityMaxPool8>(context);
}
