//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

int64_t extractMaxOutputBoxesPerClass(IE::NonMaxSuppressionOpAdaptor nms) {
    int64_t maxOutputBoxesPerClass = 0;  // default value

    if (nms.getMaxOutputBoxesPerClass() != nullptr) {
        auto maxBoxesConst = nms.getMaxOutputBoxesPerClass().getDefiningOp<Const::DeclareOp>();
        if (maxBoxesConst != nullptr && maxBoxesConst.getContentAttr().isSplat()) {
            const auto maxBoxesContent = maxBoxesConst.getContent();
            return maxBoxesContent.getSplatValue<int64_t>();
        }
    }
    if (nms.getMaxOutputBoxesPerClassValueAttr() != nullptr) {
        return nms.getMaxOutputBoxesPerClassValueAttr().getValue().getSExtValue();
    }

    return maxOutputBoxesPerClass;
}

double extractNMSAttrValue(mlir::Value constName, mlir::FloatAttr attrName) {
    double attrValue = 0.0f;
    if (constName != nullptr) {
        vpux::Const::DeclareOp attrConst = constName.getDefiningOp<Const::DeclareOp>();
        if (attrConst != nullptr && attrConst.getContentAttr().isSplat()) {
            vpux::Const::Content attrContent = attrConst.getContent();
            attrValue = attrContent.getSplatValue<float>();
        }
    } else if (attrName != nullptr) {
        attrValue = attrName.getValueAsDouble();
    }
    return attrValue;
}

}  // namespace

mlir::LogicalResult vpux::IE::NonMaxSuppressionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NonMaxSuppressionOpAdaptor nms(operands, attrs, prop);
    if (mlir::failed(nms.verify(loc))) {
        return mlir::failure();
    }

    const auto inScoresType = mlir::cast<vpux::NDTypeInterface>(nms.getInBoxScores().getType());
    const auto inScoresShapeInfo = ShapeInfo::fromNDType(inScoresType);
    const auto actualShape = inScoresShapeInfo.isDynamic() ? inScoresShapeInfo.bounds : inScoresShapeInfo.shape;
    const auto numBatches = actualShape[0];
    const auto numClasses = actualShape[1];
    const auto numBoxes = std::min(actualShape[2], extractMaxOutputBoxesPerClass(nms));
    SmallVector<int64_t> outShape{numBatches * numClasses * numBoxes, 3};
    TensorAttr outTensorAttr = nullptr;

    if (inScoresShapeInfo.isDynamic()) {
        // Handle dynamic case: use the actual shape as bound and set output shape to dynamic
        Bounds bounds(outShape);
        outTensorAttr = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), nullptr, bounds);
        outShape = SmallVector<int64_t>{mlir::ShapedType::kDynamic, 3};
    }

    const SmallVector<int64_t> validOutputsShape{1};
    auto s32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    inferredReturnShapes.emplace_back(outShape, s32Type, outTensorAttr);
    inferredReturnShapes.emplace_back(outShape, inScoresType.getElementType(), outTensorAttr);
    inferredReturnShapes.emplace_back(validOutputsShape, s32Type);

    return mlir::success();
}

namespace {

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::NonMaxSuppressionOp> {
public:
    using mlir::OpRewritePattern<IE::NonMaxSuppressionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::NonMaxSuppressionOp nmsOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::NonMaxSuppressionOp nmsOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (nmsOp.getMaxOutputBoxesPerClassValue().has_value() && nmsOp.getIouThresholdValue().has_value() &&
        nmsOp.getScoreThresholdValue().has_value() && nmsOp.getSoftNmsSigmaValue().has_value()) {
        return mlir::failure();
    }

    int64_t maxBoxesPerClassValue = extractMaxOutputBoxesPerClass(nmsOp);

    double iouThresholdValue = extractNMSAttrValue(nmsOp.getIouThreshold(), nmsOp.getIouThresholdValueAttr());

    double scoreThresholdValue = extractNMSAttrValue(nmsOp.getScoreThreshold(), nmsOp.getScoreThresholdValueAttr());

    double softNMSSigmaValue = extractNMSAttrValue(nmsOp.getSoftNmsSigma(), nmsOp.getSoftNmsSigmaValueAttr());

    rewriter.replaceOpWithNewOp<IE::NonMaxSuppressionOp>(
            nmsOp, nmsOp.getInBoxCoords(), nmsOp.getInBoxScores(), nullptr, nullptr, nullptr, nullptr,
            nmsOp.getBoxEncoding(), nmsOp.getSortResultDescending(), rewriter.getI64IntegerAttr(maxBoxesPerClassValue),
            rewriter.getF64FloatAttr(iouThresholdValue), rewriter.getF64FloatAttr(scoreThresholdValue),
            rewriter.getF64FloatAttr(softNMSSigmaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::NonMaxSuppressionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
