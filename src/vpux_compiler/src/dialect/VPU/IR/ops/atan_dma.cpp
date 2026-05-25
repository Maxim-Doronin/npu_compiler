//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

namespace {

mlir::Type getAuxiliaryBufferType(mlir::ModuleOp module) {
    constexpr int KER_WSZ = 512;  // kernel cmx workspace size [KB]
    return mlir::RankedTensorType::get({1, 1, 1, KER_WSZ * 1024}, getUInt8Type(module.getContext()));
}

}  // namespace

mlir::LogicalResult vpux::VPU::AtanDmaOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AtanDmaOpAdaptor atan(operands, attrs, prop);
    if (mlir::failed(atan.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = atan.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

void vpux::VPU::AtanDmaOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input) {
    auto loc = odsState.location;
    auto module = getModuleOp(odsBuilder);
    auto auxBufferType = getAuxiliaryBufferType(module);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, loc, auxBufferType);
    build(odsBuilder, odsState, input.getType(), input, auxBuffer, nullptr);
}

//
// SWOpInterface
//

bool vpux::VPU::AtanDmaOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/, Byte /*reservedMem*/) {
    return false;
}

bool vpux::VPU::AtanDmaOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/) {
    return false;
}

bool vpux::VPU::AtanDmaOp::supportCycleCostCalculation() {
    return false;
}

//
// AuxiliaryBufferOpInterface
//

SmallVector<mlir::OpOperand*> VPU::AtanDmaOp::getAuxiliaryBuffers() {
    return {&getAuxBufferMutable()};
}

//
// ClusteredOpInterface
//

bool vpux::VPU::AtanDmaOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering;
}

vpux::VPU::DistributionInfo vpux::VPU::AtanDmaOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

vpux::NDTypeInterface vpux::VPU::AtanDmaOp::getDistributedTypeForOpOperand(mlir::OpOperand& operand,
                                                                           bool hasExplicitDistributedAttr,
                                                                           SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<AtanDmaOp>(getOperation());
    if (operand.get() == origOp.getAuxBuffer()) {
        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr,
                                           siblingsAnalysis);
    }
    return mlir::dyn_cast<NDTypeInterface>(operand.get().getType());
}

vpux::NDTypeInterface vpux::VPU::AtanDmaOp::getDistributedTypeForOpResult(
        mlir::Value result, [[maybe_unused]] VPU::MultiClusterStrategy strategy,
        [[maybe_unused]] SiblingOpsAnalysis& siblingsAnalysis, [[maybe_unused]] bool hasExplicitDistributedAttr) {
    return mlir::dyn_cast<NDTypeInterface>(result.getType());
}
