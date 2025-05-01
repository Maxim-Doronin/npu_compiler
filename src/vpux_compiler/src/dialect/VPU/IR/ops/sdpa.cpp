//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SDPAOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SDPAOpAdaptor sdpa(operands, attrs, prop);
    if (mlir::failed(sdpa.verify(loc))) {
        return mlir::failure();
    }

    const auto inQType = mlir::cast<vpux::NDTypeInterface>(sdpa.getInputQ().getType());
    auto outputType = mlir::RankedTensorType::get(inQType.getShape(), inQType.getElementType(),
                                                  createTensorAttrFromType(inQType));
    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::SDPAOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverKernel;
}

void vpux::VPU::SDPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value inputQ,
                              ::mlir::Value inputK, ::mlir::Value inputV, ::mlir::Value inputMask,
                              ::mlir::Value dataStorage) {
    build(odsBuilder, odsState, inputQ, inputK, inputV, inputMask, dataStorage, {});
}

vpux::VPU::DistributionInfo vpux::VPU::SDPAOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::SDPAOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 6, "SDPAOp requires 5 inputs and 1 output, but the number of buffer is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::SDPAOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::SDPAOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::SDPAOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    TileInfo inKTile(getShape(getInputK()));
    TileInfo inVTile(getShape(getInputV()));
    TileInfo inMaskTile(getShape(getInputMask()));
    TileInfo dataStorageTile(getShape(getDataStorage()));
    auto inQTile = outputTile;

    inKTile.shape[Dim(Dims4D::Act::C)] = outputTile.shape[Dim(Dims4D::Act::C)];
    inKTile.shape[Dim(Dims4D::Act::N)] = outputTile.shape[Dim(Dims4D::Act::N)];
    inKTile.offsets[Dim(Dims4D::Act::C)] = outputTile.offsets[Dim(Dims4D::Act::C)];
    inKTile.offsets[Dim(Dims4D::Act::N)] = outputTile.offsets[Dim(Dims4D::Act::N)];

    inVTile.shape[Dim(Dims4D::Act::C)] = outputTile.shape[Dim(Dims4D::Act::C)];
    inVTile.shape[Dim(Dims4D::Act::N)] = outputTile.shape[Dim(Dims4D::Act::N)];
    inVTile.offsets[Dim(Dims4D::Act::C)] = outputTile.offsets[Dim(Dims4D::Act::C)];
    inVTile.offsets[Dim(Dims4D::Act::N)] = outputTile.offsets[Dim(Dims4D::Act::N)];

    dataStorageTile.shape[Dim(Dims4D::Act::C)] = outputTile.shape[Dim(Dims4D::Act::C)];
    dataStorageTile.shape[Dim(Dims4D::Act::N)] = outputTile.shape[Dim(Dims4D::Act::N)];
    dataStorageTile.offsets[Dim(Dims4D::Act::C)] = outputTile.offsets[Dim(Dims4D::Act::C)];
    dataStorageTile.offsets[Dim(Dims4D::Act::N)] = outputTile.offsets[Dim(Dims4D::Act::N)];

    return TilingInfo{{std::move(inQTile), std::move(inKTile), std::move(inVTile), std::move(inMaskTile),
                       std::move(dataStorageTile)}};
}

void vpux::VPU::SDPAOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::SDPAOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    const auto op = getOperation();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outputRank = outputType.getShape().size();
    SmallVector<int64_t> maxNumTiles;
    maxNumTiles = getMaxNumTilesWithAxesExclusion(op, /*axis:*/ {checked_cast<int64_t>(outputRank - 1)});
    return vpux::getSWLayerTilingStrategy(op, tilingMode, log, maxNumTiles);
}
