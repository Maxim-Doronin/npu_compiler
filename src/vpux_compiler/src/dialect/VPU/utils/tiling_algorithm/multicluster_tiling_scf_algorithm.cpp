//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/multicluster_tiling_scf_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/TileUsingInterface.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;
using namespace VPU;

namespace {
Shape getTilesFromStrategy(mlir::Operation* op, int64_t numClusters, VPU::MultiClusterStrategy strategy) {
    const auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    Shape tilesOnDim(outType.getRank(), 1);
    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        break;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        tilesOnDim[Dims4D::Act::C] = numClusters;
        break;
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
    case VPU::MultiClusterStrategy::HKSwitch:
        // Treat HKSwitch as SOH as first step
        // TODO: E#193460 Add broadcast capability
        tilesOnDim[Dims4D::Act::H] = numClusters;
        break;
    case VPU::MultiClusterStrategy::SplitOverWidth:
        tilesOnDim[Dims4D::Act::W] = numClusters;
        break;
    case VPU::MultiClusterStrategy::SplitOverBatch:
        tilesOnDim[Dims4D::Act::N] = numClusters;
        break;
    case VPU::MultiClusterStrategy::SplitOverGroup:
        tilesOnDim[Dims5D::Act::N] = numClusters;
        break;
    default:
        VPUX_THROW("Unsupported strategy for getTilesFromStrategy: {0}", strategy);
    }
    return tilesOnDim;
}
};  // namespace

mlir::LogicalResult MulticlusterTilingSCFAlgorithm::applyTiling(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                                Logger log) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);
    if (clusteredOp == nullptr) {
        log.trace("Op is not cluster-able.");
        return mlir::failure();
    }

    if (!clusteredOp.getMultiClusterStrategy().has_value()) {
        log.trace("Op has no multiclustering strategy assigned.");
        return mlir::failure();
    }

    const auto mcStrategy = clusteredOp.getMultiClusterStrategy().value();
    // TODO: E#193453 num clusters may be different based on the size of the tile, if MC axis is the
    // same as tiling axis
    const auto outShape = getBoundedShape(clusteredOp->getResult(0));
    const auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outShape, mcStrategy);
    const auto strategy = getTilesFromStrategy(operation, numClusters, mcStrategy);

    // Workaround to re-use current infrastructure from tiling.
    // TODO: E#192457 Implement proper multiclustering
    builder.modifyOpInPlace(operation, [&]() {
        operation->removeAttr(multiClusterStrategy);
        operation->setAttr(tilingStrategy, getIntArrayAttr(builder, strategy));
    });

    mlir::scf::SCFTilingOptions tilingOptions;

    const auto mcAxis = VPU::getDistributedTilingAxis(strategy.raw());
    const auto tileSizeComputationFnc = [&](mlir::OpBuilder&, mlir::Operation*) {
        const auto outShape = getShape(operation->getResult(0));
        if (outShape.isDynamic()) {
            return dynamicTileSizeComputation(builder, {operation}, nullptr, strategy,
                                              outShape[Dim(mcAxis)] != mlir::ShapedType::kDynamic);
        }

        std::unordered_map<Dim, std::pair<int64_t, int64_t>> emptyRemainders;

        return staticTileSizeComputation(builder, {operation}, nullptr, strategy, getShape(operation->getResult(0)),
                                         emptyRemainders);
    };

    tilingOptions.setTileSizeComputationFunction(tileSizeComputationFnc);
    tilingOptions.setLoopType(mlir::scf::SCFTilingOptions::LoopType::ForallOp);

    auto tilingResult = mlir::scf::tileUsingSCF(builder, mlir::cast<mlir::TilingInterface>(operation), tilingOptions);
    if (mlir::failed(tilingResult) || tilingResult->loops.empty()) {
        return operation->emitError("Tiling algorithm failed");
    }

    // E-162999 rewrite to update order attribute for output types more elegantly
    // tileUsingSCF drops the output order in the ForAllOp and terminator. This adds it back.
    auto outputType = operation->getResult(0).getType();
    llvm::for_each(tilingResult->loops, [&](mlir::LoopLikeOpInterface loop) {
        auto forallOp = mlir::cast<mlir::scf::ForallOp>(loop.getOperation());
        forallOp.getResult(0).setType(outputType);

        auto* terminator = forallOp.getBody()->getTerminator();
        if (auto inParallelOp = mlir::dyn_cast_or_null<mlir::scf::InParallelOp>(terminator)) {
            auto parallelInsertSliceOps = inParallelOp.getOps<mlir::tensor::ParallelInsertSliceOp>();
            for (auto insertOp : parallelInsertSliceOps) {
                insertOp.getDestMutable().get().setType(outputType);
                if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertOp.getDest())) {
                    auto argIndex = blockArg.getArgNumber() - forallOp.getInductionVars().size();
                    forallOp.getOutputs()[argIndex].setType(outputType);
                }
            }
        }
    });

    builder.replaceOp(operation, tilingResult->replacements);

    return mlir::success();
}

SmallVector<mlir::Operation*> MulticlusterTilingSCFAlgorithm::applySCFTilingAndFusion(mlir::Operation* /*operation*/,
                                                                                      mlir::RewriterBase& /*builder*/,
                                                                                      Logger log) {
    log.trace("MC fusion is not yet implemented.");
    return {};
}
