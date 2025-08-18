//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/custom_float.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::ClampOp::verify() {
    auto inElemType = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getElementType();
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inElemType)) {
        return errorAt(*this, "Per-axis quantized type is not supported. Got: {0}", inElemType);
    }

    const auto minVal = getMinAttr().getValueAsDouble();
    const auto maxVal = getMaxAttr().getValueAsDouble();
    if (minVal > maxVal) {
        return errorAt(*this, "ClampOp {0} has invalid minAttr {1} and maxAttr {2}", getLoc(), minVal, maxVal);
    }

    return mlir::success();
}

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ClampOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ClampOpAdaptor clamp(operands, attrs, prop);
    if (mlir::failed(clamp.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(clamp.getInput().getType());
    const auto outDesc = vpux::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

namespace {

//
// Convert Attr to FP16
//

class ConvertAttrToFP16 final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    using mlir::OpRewritePattern<IE::ClampOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertAttrToFP16::matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto minVal = origOp.getMinAttr().getValueAsDouble();
    const auto maxVal = origOp.getMaxAttr().getValueAsDouble();

    // There is a case when a Clamp operation has default min or max values.
    // They are set to the numeric limits for FP32. But the NPU only supports FP16 precision.
    const auto isOutOfFP16Range = [](double value) {
        return std::abs(value) > std::numeric_limits<vpux::type::float16>::max();
    };
    if (!isOutOfFP16Range(minVal) && !isOutOfFP16Range(maxVal)) {
        return mlir::failure();
    }

    const auto lowestF16 = checked_cast<double>(std::numeric_limits<vpux::type::float16>::lowest());
    const auto maxF16 = checked_cast<double>(std::numeric_limits<vpux::type::float16>::max());
    if (minVal > maxF16 || maxVal < lowestF16) {
        Logger::global().warning("ClampOp operation at location {0} has a value range from {1} to {2}, which exceeds "
                                 "the FP16 data range. The output values are adjusted to fp16_lowest or fp16_max.",
                                 origOp.getLoc(), minVal, maxVal);
    }
    const auto newMin = std::clamp(minVal, lowestF16, maxF16);
    const auto newMax = std::clamp(maxVal, lowestF16, maxF16);
    const auto minAttr = getFPAttr(origOp.getContext(), newMin);
    const auto maxAttr = getFPAttr(origOp.getContext(), newMax);

    rewriter.replaceOpWithNewOp<IE::ClampOp>(origOp, origOp.getInput(), minAttr, maxAttr);
    return mlir::success();
}

//
// Fuse Clamps
//

class FuseClamps final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    using mlir::OpRewritePattern<IE::ClampOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseClamps::matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const {
    auto parentOp = origOp.getInput().getDefiningOp<IE::ClampOp>();
    if (parentOp == nullptr) {
        return mlir::failure();
    }

    if (!parentOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    const auto minParentOp = parentOp.getMinAttr().getValueAsDouble();
    const auto minOrigOp = origOp.getMinAttr().getValueAsDouble();
    const auto maxParentOp = parentOp.getMaxAttr().getValueAsDouble();
    const auto maxOrigOp = origOp.getMaxAttr().getValueAsDouble();

    const auto newMin = std::max(minParentOp, minOrigOp);
    const auto newMax = std::min(maxParentOp, maxOrigOp);

    rewriter.replaceOpWithNewOp<IE::ClampOp>(origOp, parentOp.getInput(), getFPAttr(rewriter, newMin),
                                             getFPAttr(rewriter, newMax));
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ClampOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseClamps>(ctx);
    patterns.add<ConvertAttrToFP16>(ctx);
}
