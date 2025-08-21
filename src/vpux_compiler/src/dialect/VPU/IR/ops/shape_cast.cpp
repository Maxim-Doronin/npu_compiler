//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;
using namespace VPU;
namespace {
DimArr getReshapedDims(ShapeCastOp shapeCastOp) {
    auto inputShape = getShape(shapeCastOp.getInput());
    auto outputShape = getShape(shapeCastOp.getOutput());
    DimArr reshapedDims;
    for (auto i : irange(inputShape.size())) {
        Dim dim(i);
        if (inputShape[dim] != outputShape[dim]) {
            reshapedDims.push_back(dim);
        }
    }
    return reshapedDims;
}
}  // namespace

mlir::LogicalResult vpux::VPU::ShapeCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ShapeCastOpAdaptor shapeCast(operands, attrs, prop);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(shapeCast.getShape());
    const auto inType = mlir::cast<vpux::NDTypeInterface>(shapeCast.getInput().getType());

    auto getDistType = [&](VPU::DistributedTypeInterface inDistInterface) {
        const auto arch = config::getArch(mlir::isa<mlir::BlockArgument>(operands[0])
                                                  ? operands[0].getParentRegion()->getParentOfType<mlir::ModuleOp>()
                                                  : operands[0].getDefiningOp());
        const auto distAttr =
                VPUIP::getDistributedAttrAfterShapeCast<VPU::DistributedTensorType>(inDistInterface, outShape, arch);
        return inDistInterface.changeShapeForExplicitDistribution(ShapeRef(outShape), distAttr);
    };

    vpux::NDTypeInterface outType;
    const auto distributedIn = mlir::dyn_cast<VPU::DistributedTypeInterface>(inType);
    if (distributedIn != nullptr && distributedIn.containsDistributedTypes()) {
        outType = getDistType(distributedIn);
    } else {
        outType = inType.changeShape(ShapeRef(outShape));
    }

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingViewLikeOpInterface
//

vpux::InputTiling vpux::VPU::ShapeCastOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    VPUX_THROW_UNLESS(isSupportedOutTile(outputTile),
                      "ShapeCastOp does not support out tile with shape {0}, offset {1}", outputTile.shape,
                      outputTile.offsets);

    const auto inputShape = vpux::getShape(getInput());
    const auto outputShape = vpux::getShape(getOutput());
    const auto tilingDims = getNonOneDim(outputTile.axis);

    TileInfo inputTile(inputShape);
    if (tilingDims.empty()) {
        return InputTiling{inputTile};
    }

    auto reshapedDims = getReshapedDims(*this);

    const auto tilingDim = tilingDims.front();
    const auto tilingDimIsReshaped = llvm::find(reshapedDims, tilingDim) != reshapedDims.end();
    if (tilingDimIsReshaped) {
        VPUX_THROW_WHEN(outputShape[tilingDim] == 0, "Invalid output shape {0}", outputShape);
        inputTile.shape[tilingDim] = outputTile.shape[tilingDim] * inputShape[tilingDim] / outputShape[tilingDim];
        inputTile.offsets[tilingDim] = outputTile.offsets[tilingDim] * inputShape[tilingDim] / outputShape[tilingDim];
    } else {
        inputTile.shape[tilingDim] = outputTile.shape[tilingDim];
        inputTile.offsets[tilingDim] = outputTile.offsets[tilingDim];
    }

    return InputTiling{inputTile};
}

void vpux::VPU::ShapeCastOp::adjustAttrs(const TilingInfo&, const TileInfo& outputTile, ShapeRef) {
    auto newShape = getIntArrayAttr(getContext(), outputTile.shape);
    setShapeAttr(newShape);
}

bool vpux::VPU::ShapeCastOp::isSupportedTilingDim(DimArrRef tilingDims) {
    if (tilingDims.size() > 1) {
        return false;
    }
    if (tilingDims.empty()) {
        return true;
    }

    auto reshapedDims = getReshapedDims(*this);
    // Only support shape cast scenarios where exactly two adjacent dimensions are reshaped
    if (reshapedDims.size() != 2) {
        return false;
    }

    auto tilingDim = tilingDims.front();
    auto dimOrder = DimsOrder::fromValue(getInput());
    auto idx0 = checked_cast<int64_t>(dimOrder.dimPos(reshapedDims[0]));
    auto idx1 = checked_cast<int64_t>(dimOrder.dimPos(reshapedDims[1]));
    if (std::abs(idx0 - idx1) != 1) {
        return false;
    }
    auto highestReshapedDim = *std::min_element(reshapedDims.begin(), reshapedDims.end(), [&](Dim a, Dim b) {
        return dimOrder.dimPos(a) < dimOrder.dimPos(b);
    });
    return dimOrder.dimPos(tilingDim) <= dimOrder.dimPos(highestReshapedDim);
}

bool vpux::VPU::ShapeCastOp::isSupportedOutTile(const TileInfo& outTile) {
    auto tilingDims = getNonOneDim(Shape(outTile.axis));
    if (!isSupportedTilingDim(tilingDims)) {
        return false;
    }
    if (tilingDims.empty()) {
        return true;
    }

    auto inputShape = vpux::getShape(getInput());
    auto outputShape = vpux::getShape(getOutput());
    auto reshapedDims = getReshapedDims(*this);
    auto tilingDim = tilingDims.front();

    const auto tilingDimIsReshaped = llvm::find(reshapedDims, tilingDim) != reshapedDims.end();
    if (!tilingDimIsReshaped) {
        return true;
    }

    VPUX_THROW_WHEN(outputShape[tilingDim] == 0, "Invalid output shape {0}", outputShape);
    if ((outTile.shape[tilingDim] * inputShape[tilingDim]) % outputShape[tilingDim] != 0) {
        return false;
    }
    return (outTile.offsets[tilingDim] * inputShape[tilingDim]) % outputShape[tilingDim] == 0;
}

//
// DistributedCastOpInterface
//
// This function infers the type and distribution of the output tensor based on the input type and distribution.
// 1. For duplicated mode, we can always get the casted distributed type by changing the output shape
// 2. For segmented like mode, need to check if the tiling dimension is supported or not.

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>> vpux::VPU::ShapeCastOp::inferCastedTypeAndDistribution(
        vpux::NDTypeInterface inType, VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }

    const auto srcShape = inType.getShape();
    const auto dstType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outShape = dstType.getShape();

    auto mode = distribution.getDistributionMode();
    if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
        auto tilingDim = Dim(getDistributedTilingAxis(distribution.getNumTiles()));
        if (!isSupportedTilingDim({tilingDim})) {
            return mlir::failure();
        }
    }

    if (srcShape.size() != outShape.size()) {
        return mlir::failure();
    }

    if (auto sparseTensor = mlir::dyn_cast<VPU::SparseTensorType>(inType)) {
        return mlir::failure();
    }

    if (!VPU::isDistributionWithExplicitShapesAndOffsets(distribution)) {
        const auto typeComponents = TypeComponents().setShape(outShape);
        return std::make_pair(mlir::cast<mlir::Type>(inType.changeTypeComponents(typeComponents)), distribution);
    }

    const auto reshapedDims = getReshapedDims(*this);
    auto reshapePerClusterShape = [&](ArrayRef<SmallVector<int64_t>> perClusterShapes) {
        SmallVector<SmallVector<int64_t>> reshapedShapes(perClusterShapes.begin(), perClusterShapes.end());
        for (auto& shape : reshapedShapes) {
            for (auto dim : reshapedDims) {
                shape[dim.ind()] = outShape[dim];
            }
        }
        return reshapedShapes;
    };

    auto outPerClusterMemShapes = reshapePerClusterShape(distribution.getMemoryShapes());
    auto outPerClusterComputeShapes = reshapePerClusterShape(distribution.getComputeShapes());

    auto outDistribution = distribution;
    outDistribution.setMemoryShapes(outPerClusterMemShapes);
    outDistribution.setComputeShapes(outPerClusterComputeShapes);
    const auto typeComponents = TypeComponents().setShape(outShape);
    return std::make_pair(mlir::cast<mlir::Type>(inType.changeTypeComponents(typeComponents)), outDistribution);
}

mlir::OpFoldResult vpux::VPU::ShapeCastOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    if (inputType == outputType) {
        return getInput();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());
    if (mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputType.getElementType())) {
        return nullptr;
    }
    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return static_cast<Const::ContentAttr>(attr).transform().reshape(outputType.getShape()).get();
    }

    return nullptr;
}

//
// FuseShapeCast
//

namespace {
class FuseShapeCast final : public mlir::OpRewritePattern<VPU::ShapeCastOp> {
public:
    using mlir::OpRewritePattern<VPU::ShapeCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseShapeCast::matchAndRewrite(VPU::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp<VPU::ShapeCastOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::ShapeCastOp>(origOp, prevOp.getInput(), origOp.getShape());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::ShapeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseShapeCast>(ctx);
}
