//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "mlir/IR/Attributes.h"

using namespace vpux::VPU;

mlir::OpFoldResult vpux::VPU::getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim) {
    const auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());
    if (!outputType.hasStaticShape() && !operation->getOperand(0).hasOneUse()) {
        auto dimUser = llvm::find_if(operation->getOperand(0).getUsers(), [](auto* user) {
            return mlir::isa<mlir::tensor::DimOp>(user);
        });

        if (dimUser != operation->getOperand(0).getUsers().end()) {
            return dimUser->getResult(0);
        }
    }

    mlir::ReifiedRankedShapedTypeDims resultShape;
    if (mlir::failed(reifyResultShapes(builder, operation, resultShape))) {
        return builder.getIndexAttr(outputType.getDimSize(dim));
    }

    return resultShape[0][dim];
}

mlir::Value vpux::VPU::generateTile(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value origInput,
                                    const SCFTileInfo& inputTileInfo) {
    auto origType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());

    auto staticNewShape = mlir::getConstantIntValues(inputTileInfo.shape);
    if (origType.getShape().isStatic() && staticNewShape.has_value() &&
        llvm::equal(origType.getShape().raw(), staticNewShape.value())) {
        return origInput;
    }

    SmallVector<mlir::OpFoldResult> defaultStrides(inputTileInfo.offsets.size(), builder.getIndexAttr(1));

    auto extractTile = builder.create<mlir::tensor::ExtractSliceOp>(
            appendLoc(loc, "extractSlice"), origInput, inputTileInfo.offsets, inputTileInfo.shape, defaultStrides);

    auto newShape = getShape(extractTile.getResult());
    auto newType = origType.changeShape(ShapeRef(newShape));
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(newType)) {
        newType = boundedType.changeBounds(inputTileInfo.bounds);
    }

    // by default output type loses NPU-specific attributes so we have to set it manually
    extractTile->getResult(0).setType(newType);

    return extractTile;
}

mlir::Type vpux::VPU::extractResultType(mlir::Type origType, SCFShapeRef newShape, BoundsRef bounds) {
    auto ndTensorType = mlir::cast<vpux::NDTypeInterface>(origType);
    auto origElemType = ndTensorType.getElementType();

    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(origElemType),
                    "Per axis quantized types are not supported in scf");

    const auto tensorDesc =
            vpux::getTensorAttr(origElemType.getContext(), ndTensorType.getDimsOrder(), ndTensorType.getMemSpace(),
                                mlir::isa<Core::BoundedTensorType>(origType) ? bounds : Bounds{});

    SmallVector<mlir::Value> dynamicDims;  // unused cause for shape static dims are enough
    SmallVector<int64_t> staticDims;
    mlir::dispatchIndexOpFoldResults(newShape, dynamicDims, staticDims);
    return mlir::RankedTensorType::get(staticDims, origElemType, tensorDesc);
}

SCFTileInfo vpux::VPU::getWeightsTableSCFTile(mlir::Type origWeightsTableType, mlir::OpBuilder& builder,
                                              const SCFTileInfo& outputTile) {
    auto origWeightsTableShape = mlir::cast<mlir::ShapedType>(origWeightsTableType).getShape();

    SCFTileInfo weightsTableTile(origWeightsTableShape, builder);
    weightsTableTile.offsets[0] = outputTile.offsets[Dims4D::Act::C.ind()];
    weightsTableTile.shape[0] = outputTile.shape[Dims4D::Act::C.ind()];
    return weightsTableTile;
}

std::pair<std::optional<mlir::Range>, std::optional<int64_t>> vpux::VPU::solutionForOutputRange(
        mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile, Dim dim, const int64_t kernel,
        const int64_t stride, const int64_t origInputSize, const std::pair<int64_t, int64_t>& origPadding,
        mlir::OpFoldResult& padBefore, mlir::OpFoldResult& padAfter) {
    auto zero = builder.getIndexAttr(0);
    auto one = builder.getIndexAttr(1);
    mlir::Range inputRange = {zero, zero, one};
    mlir::Range outputRange = {outputTile.offsets[dim.ind()], outputTile.shape[dim.ind()], one};

    // define dimensions (d0, d1, ...) as variables which are represented by loop dim identifier
    // and symbols (s0, s1, ...) which are either known constants or known attributes of operation
    mlir::AffineExpr s0, s1, d0, d1;
    bindDims(builder.getContext(), d0, d1);
    bindSymbols(builder.getContext(), s0, s1);

    mlir::AffineExpr sizeExpr = (d0 - 1) * stride + kernel - origPadding.first;
    auto sizeMap = mlir::AffineMap::get(1, 0, {sizeExpr}, builder.getContext());

    std::optional<int64_t> dimBound;
    if (!outputTile.bounds.raw().empty()) {
        auto outputTileBound = builder.getIntegerAttr(builder.getIndexType(), outputTile.bounds[dim]);
        SmallVector<mlir::Attribute> resultsAttrs;
        if (sizeMap.constantFold({outputTileBound}, resultsAttrs).succeeded()) {
            if (auto result = mlir::dyn_cast<mlir::IntegerAttr>(resultsAttrs.front())) {
                dimBound = result.getInt() - origPadding.second;
            }
        }
    }

    if (mlir::isConstantIntValue(outputTile.axis[dim.ind()], 1)) {
        return {std::nullopt, dimBound};
    }

    const auto hasPadBefore = origPadding.first != 0;
    const auto hasPadAfter = origPadding.second != 0;

    // input offset is based on output tile offset and operation's parameters
    // current calculation is
    // offset: max((output offset) * stride - padding, 0).
    // size: (output size - 1) * stride + kernel - padding
    // if operation has padding, the median tile size will be corrected later if needed
    if (!hasPadBefore && stride == 1) {
        inputRange.offset = outputRange.offset;
    } else {
        mlir::AffineExpr offsetExpr = d0 * stride - origPadding.first;
        auto offsetMap = mlir::AffineMap::get(1, 1, {offsetExpr, s0}, builder.getContext());
        inputRange.offset = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "inputOffset"), offsetMap,
                                                                      {outputRange.offset, zero});

        auto minDiffMap = mlir::AffineMap::get(1, 2, {s0 - offsetExpr, s1}, builder.getContext());
        auto minDiffValue = mlir::affine::makeComposedFoldedAffineMin(builder, appendLoc(loc, "minDiff"), minDiffMap,
                                                                      {outputRange.offset, zero, zero});
        auto padBeforeMap = mlir::AffineMap::get(0, 2, {s0, s1}, builder.getContext());
        padBefore = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "paddingBefore"), padBeforeMap,
                                                              {minDiffValue, builder.getIndexAttr(origPadding.first)});
    }

    inputRange.size = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "inputSize"), sizeMap,
                                                                  {outputRange.size});
    if (hasPadAfter) {
        auto minDiffMap = mlir::AffineMap::get(2, 2, {d1 + sizeExpr - s0, s1}, builder.getContext());
        auto minDiffValue = mlir::affine::makeComposedFoldedAffineMin(
                builder, appendLoc(loc, "minDiff"), minDiffMap,
                {outputRange.offset, inputRange.offset, builder.getIndexAttr(origInputSize), zero});
        auto padAfterMap = mlir::AffineMap::get(0, 2, {s0, s1}, builder.getContext());
        padAfter = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "paddingAfter"), padAfterMap,
                                                             {minDiffValue, builder.getIndexAttr(origPadding.second)});
    }

    return {inputRange, dimBound};
}

bool vpux::VPU::checkFusion(mlir::OpOperand& consumer, mlir::OpResult producerCandidate) {
    // TODO E-172888 rewrite unified code for checking compatibility with current VF

    if (!mlir::isa<mlir::TilingInterface>(producerCandidate.getOwner())) {
        return false;
    }

    if (VPU::isPureViewOp(producerCandidate.getOwner()) || VPU::isPureViewOp(consumer.getOwner())) {
        return true;
    }

    const auto hasMCStategy = [](mlir::Operation* operation) {
        auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);
        return clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value();
    };

    auto consumerHasStrategy = hasMCStategy(consumer.getOwner());
    auto producerHasStrategy = hasMCStategy(producerCandidate.getOwner());

    if (!consumerHasStrategy && !producerHasStrategy) {
        return true;
    }

    if (consumerHasStrategy ^ producerHasStrategy) {
        return false;
    }

    auto producerClusterOp = mlir::cast<VPU::ClusteredOpInterface>(producerCandidate.getOwner());
    auto consumerClusterOp = mlir::cast<VPU::ClusteredOpInterface>(consumer.getOwner());

    VPU::SiblingOpsAnalysis siblingAnalisys(consumer.getOwner());

    auto consumerDistrType = mlir::cast<VPU::DistributedTensorType>(
            consumerClusterOp.getDistributedTypeForOpOperand(consumer, false, siblingAnalisys));
    auto producerDistrType = mlir::cast<VPU::DistributedTensorType>(producerClusterOp.getDistributedTypeForOpResult(
            producerCandidate, producerClusterOp.getMultiClusterStrategy().value(), siblingAnalisys, false));

    return areDistributionAttrsCompatible(producerDistrType, consumerDistrType, true).succeeded();
}
