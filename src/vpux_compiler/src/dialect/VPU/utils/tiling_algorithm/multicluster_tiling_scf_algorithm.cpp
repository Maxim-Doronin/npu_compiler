//
// Copyright (C) 2025-2026 Intel Corporation
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

    // Clustering replicates data across all clusters without splitting dimension.
    // Generate scf.forall with numClusters iterations where each iteration runs the
    // full operation. This representation is needed for future vertical fusion support.
    // To avoid UB from overlapping parallel_insert_slice writes (all iterations writing
    // to [0,0,0,0]), expand dim 0 by numClusters so each iteration writes a disjoint
    // slice. After the forall, extract_slice recovers the original shape.
    if (mcStrategy == VPU::MultiClusterStrategy::Clustering) {
        log.trace("Generating scf.forall for Clustering strategy with {0} clusters.", numClusters);

        builder.setInsertionPoint(operation);

        auto loc = operation->getLoc();
        const auto numResults = operation->getNumResults();
        constexpr int64_t expandDimIdx = 0;

        // Compute original output sizes and create expanded tensor.empty for each result.
        // This is similar to tensor::getOrCreateDestinations, but we need to preserve
        // the encoding (e.g., layout order, bounds) when creating tensor.empty.
        // TODO: E-210745 Extend tensor::getOrCreateDestinations to Preserve Tensor Encoding.
        mlir::ReifiedRankedShapedTypeDims reifiedShapes;
        const bool hasReified = mlir::succeeded(mlir::reifyResultShapes(builder, operation, reifiedShapes));

        SmallVector<mlir::Value> outputEmpties;
        SmallVector<SmallVector<mlir::OpFoldResult>> allOrigSizes;
        outputEmpties.reserve(numResults);
        allOrigSizes.reserve(numResults);

        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto resultType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());

            SmallVector<mlir::OpFoldResult> origSizes;
            if (resultType.hasStaticShape()) {
                for (auto dim : llvm::seq<int64_t>(0, resultType.getRank())) {
                    origSizes.push_back(builder.getIndexAttr(resultType.getDimSize(dim)));
                }
            } else if (hasReified) {
                origSizes = reifiedShapes[idx];
            } else {
                return operation->emitError("Clustering requires static shapes or reifyResultShapes support");
            }

            // Expand dim 0: origDim0 * numClusters.
            // Use origSizes[0] (the already-built OpFoldResult) so dynamic sizes are handled
            // correctly. If static, fold to a constant; otherwise emit arith.muli.
            SmallVector<mlir::OpFoldResult> expandedSizes = origSizes;
            if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(origSizes[expandDimIdx])) {
                auto dim0 = mlir::dyn_cast<mlir::IntegerAttr>(attr).getInt();
                expandedSizes[expandDimIdx] = builder.getIndexAttr(dim0 * numClusters);
            } else {
                auto dim0Val = mlir::getValueOrCreateConstantIndexOp(builder, loc, origSizes[expandDimIdx]);
                auto numClustersVal = builder.create<mlir::arith::ConstantIndexOp>(loc, numClusters);
                expandedSizes[expandDimIdx] =
                        builder.create<mlir::arith::MulIOp>(loc, dim0Val, numClustersVal).getResult();
            }

            // Build expanded encoding: if the original has bounds, update dim 0 bound.
            auto expandedEncoding = resultType.getEncoding();
            if (auto tensorAttr = mlir::dyn_cast_if_present<vpux::TensorAttr>(expandedEncoding)) {
                auto origBounds = tensorAttr.getBounds();
                if (!origBounds.empty()) {
                    auto expandedBounds = Bounds(origBounds.raw());
                    expandedBounds[Dim(expandDimIdx)] = origBounds[Dim(expandDimIdx)] * numClusters;
                    expandedEncoding =
                            vpux::getTensorAttr(builder.getContext(), tensorAttr.getOrder(), tensorAttr.getMemSpace(),
                                                expandedBounds, tensorAttr.getDynamicDimsMask());
                }
            }

            auto emptyOp = builder.create<mlir::tensor::EmptyOp>(loc, expandedSizes, resultType.getElementType(),
                                                                 expandedEncoding);
            outputEmpties.push_back(emptyOp.getResult());
            allOrigSizes.push_back(std::move(origSizes));
        }

        builder.modifyOpInPlace(operation, [&] {
            operation->removeAttr(multiClusterStrategy);
        });

        // Create scf.forall with numClusters iterations (normalized loop: only upper bounds)
        SmallVector<mlir::OpFoldResult> upperBounds = {builder.getIndexAttr(numClusters)};

        auto forallOp = builder.create<mlir::scf::ForallOp>(loc, upperBounds,
                                                            /*outputs=*/mlir::ValueRange{outputEmpties},
                                                            /*mapping=*/std::nullopt);

        auto* terminator = forallOp.getBody()->getTerminator();
        builder.setInsertionPoint(terminator);

        // Compute per-cluster offset in dim 0: iv * origDim0.
        // Derive step from allOrigSizes (the pre-built OpFoldResults) so dynamic dim0 is correct.
        auto iv = forallOp.getBody()->getArgument(0);
        SmallVector<mlir::Value> dim0Offsets;
        dim0Offsets.reserve(numResults);
        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto origDim0OFR = allOrigSizes[idx][expandDimIdx];
            // Fast path: static size 1 — offset is just the induction variable.
            if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(origDim0OFR)) {
                if (mlir::dyn_cast<mlir::IntegerAttr>(attr).getInt() == 1) {
                    dim0Offsets.push_back(iv);
                    continue;
                }
            }
            auto step = mlir::getValueOrCreateConstantIndexOp(builder, loc, origDim0OFR);
            dim0Offsets.push_back(builder.create<mlir::arith::MulIOp>(loc, iv, step).getResult());
        }

        // Clone the original operation inside the forall
        auto* clonedOp = builder.clone(*operation);

        // Create parallel_insert_slice for each result — disjoint writes along dim 0
        auto inParallelOp = mlir::cast<mlir::scf::InParallelOp>(terminator);
        builder.setInsertionPointToStart(inParallelOp.getBody());

        auto regionOutArgs = forallOp.getRegionOutArgs();
        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto resultType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());
            auto rank = resultType.getRank();
            SmallVector<mlir::OpFoldResult> offsets(rank, builder.getIndexAttr(0));
            offsets[expandDimIdx] = dim0Offsets[idx];
            SmallVector<mlir::OpFoldResult> strides(rank, builder.getIndexAttr(1));

            builder.create<mlir::tensor::ParallelInsertSliceOp>(loc, clonedOp->getResult(idx), regionOutArgs[idx],
                                                                offsets, allOrigSizes[idx], strides);
        }

        // Fix forall output types to include encoding from expanded shape.
        // For dynamic dim0 keep kDynamic; the runtime size comes from the tensor.empty operand.
        // Update bounds in encoding to match expanded dim 0.
        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto origType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());
            auto expandedShape = llvm::to_vector(origType.getShape());
            if (expandedShape[expandDimIdx] != mlir::ShapedType::kDynamic) {
                expandedShape[expandDimIdx] *= numClusters;
            }
            auto expandedEncoding = origType.getEncoding();
            if (auto tensorAttr = mlir::dyn_cast_if_present<vpux::TensorAttr>(expandedEncoding)) {
                auto origBounds = tensorAttr.getBounds();
                if (!origBounds.empty()) {
                    auto expandedBounds = Bounds(origBounds.raw());
                    expandedBounds[Dim(expandDimIdx)] = origBounds[Dim(expandDimIdx)] * numClusters;
                    expandedEncoding =
                            vpux::getTensorAttr(builder.getContext(), tensorAttr.getOrder(), tensorAttr.getMemSpace(),
                                                expandedBounds, tensorAttr.getDynamicDimsMask());
                }
            }
            auto expandedType = mlir::RankedTensorType::get(expandedShape, origType.getElementType(), expandedEncoding);
            forallOp.getResult(idx).setType(expandedType);
            regionOutArgs[idx].setType(expandedType);
        }

        // Extract one copy from each expanded result to recover original shape
        builder.setInsertionPointAfter(forallOp);
        SmallVector<mlir::Value> replacements;
        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto origType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());
            auto rank = origType.getRank();
            SmallVector<mlir::OpFoldResult> offsets(rank, builder.getIndexAttr(0));
            SmallVector<mlir::OpFoldResult> strides(rank, builder.getIndexAttr(1));

            auto extractOp = builder.create<mlir::tensor::ExtractSliceOp>(loc, origType, forallOp.getResult(idx),
                                                                          offsets, allOrigSizes[idx], strides);
            replacements.push_back(extractOp.getResult());
        }
        builder.replaceOp(operation, replacements);
        return mlir::success();
    }

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
    // Handle all results for multi-output ops (TopK, LSTMGates).
    llvm::for_each(tilingResult->loops, [&](mlir::LoopLikeOpInterface loop) {
        auto forallOp = mlir::cast<mlir::scf::ForallOp>(loop.getOperation());
        for (unsigned idx = 0; idx < operation->getNumResults(); ++idx) {
            forallOp.getResult(idx).setType(operation->getResult(idx).getType());
        }

        auto* terminator = forallOp.getBody()->getTerminator();
        if (auto inParallelOp = mlir::dyn_cast_or_null<mlir::scf::InParallelOp>(terminator)) {
            auto parallelInsertSliceOps = inParallelOp.getOps<mlir::tensor::ParallelInsertSliceOp>();
            for (auto insertOp : parallelInsertSliceOps) {
                if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertOp.getDest())) {
                    auto argIndex = blockArg.getArgNumber() - forallOp.getInductionVars().size();
                    auto outputType = operation->getResult(argIndex).getType();
                    insertOp.getDestMutable().get().setType(outputType);
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
