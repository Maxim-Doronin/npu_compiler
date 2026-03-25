//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"

#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReshapeOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReshapeOpAdaptor reshape(operands, attrs, prop);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(reshape.getShapeValue());

    const auto inType = mlir::cast<vpux::NDTypeInterface>(reshape.getInput().getType());

    const auto typeComponents =
            TypeComponents().setShape(ShapeRef(outShape)).setDimsOrder(DimsOrder::fromNumDims(outShape.size()));
    auto outType = inType.changeTypeComponents(typeComponents);

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::ReshapeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return static_cast<Const::ContentAttr>(attr).transform().reshape(vpux::getShape(getOutput())).get();
    }

    return nullptr;
}

//
// ConvertToShapeCast
//

namespace {

class ConvertToShapeCast final : public mlir::OpRewritePattern<VPU::ReshapeOp> {
public:
    using mlir::OpRewritePattern<VPU::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToShapeCast::matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    if (!inputType.getDimsOrder().isIdentity() || inputType.getRank() != outputType.getRank()) {
        return mlir::failure();
    }

    auto shapeValueAttr = origOp.getShapeValueAttr();
    if (shapeValueAttr == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::ShapeCastOp>(origOp, origOp.getInput(), origOp.getShapeValueAttr());
    return mlir::success();
}

}  // namespace

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<VPU::ReshapeOp> {
public:
    using mlir::OpRewritePattern<VPU::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp();
    if (prevOp == nullptr || !prevOp->hasOneUse()) {
        return mlir::failure();
    }
    if (!mlir::isa<VPU::SqueezeOp, VPU::UnsqueezeOp, VPU::ReshapeOp, VPU::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = outputType.getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    auto newOp = rewriter.replaceOpWithNewOp<VPU::ReshapeOp>(origOp, prevOp->getOperand(0), outputShapeAttr);
    extendOpLoc(newOp, "fused_with_other");

    return mlir::success();
}

}  // namespace

//
// ConvertToAffineReshape
//

namespace {

class ConvertToAffineReshape final : public mlir::OpRewritePattern<VPU::ReshapeOp> {
public:
    using mlir::OpRewritePattern<VPU::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToAffineReshape::matchAndRewrite(VPU::ReshapeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = outputType.getShape();
    const auto outShapeAttr = getIntArrayAttr(getContext(), outputShape);

    const auto inShape = inputType.getShape();
    const auto reassociationMap = vpux::IE::getReassociationMap(inShape, outputShape);
    if (mlir::failed(reassociationMap)) {
        return mlir::failure();
    }

    // If no valid output layout can be inferred, don't replace with AffineReshape
    auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outputLayout = Const::inferAffineReshapeOutputLayout(
            inOrder.toPermutation(), getIntArrayOfArray(getContext(), reassociationMap.value()));
    if (!outputLayout.has_value()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(
            origOp, origOp.getInput(), getIntArrayOfArray(getContext(), reassociationMap.value()), outShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseReshapes>(ctx);
    patterns.add<ConvertToShapeCast>(ctx);
    patterns.add<ConvertToAffineReshape>(ctx);
}
