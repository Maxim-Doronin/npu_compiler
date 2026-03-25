//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/VPU/utils/reorder_ir_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_pass_config_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "vpux/utils/logger/logger.hpp"

#include <cmath>

using namespace vpux;

/*
    Correct offsets and sizes of slice operation based on remainders information
    Remainders map contains information about every dimension that requires correction
    First element of pair is a bound until which we have current step, second element is remainder step size
    Correction is applied only for computed sizes.
    For every dimension with correction requirement we expect to have:
        - offset represented by block argument of loop operation
        - dynamic size represented by affine min operation defined over loop induction variable
    Correction is applied only if both conditions are met
    New offset is computed as:
        offset = ((inductionVar - mainStepBound) floorDiv stepSize) * loopStep + mainStepBound
*/
void correctOffsetAndSizeByRemainder(mlir::RewriterBase& builder, mlir::OffsetSizeAndStrideOpInterface slice,
                                     const std::unordered_map<Dim, std::pair<int64_t, int64_t>>& remainders) {
    if (remainders.empty()) {
        return;
    }

    auto offsets = slice.getMixedOffsets();
    auto sizes = slice.getMixedSizes();

    for (auto& [dim, data] : remainders) {
        if (!slice.isDynamicOffset(dim.ind())) {
            continue;
        }

        VPUX_THROW_UNLESS(slice.isDynamicSize(dim.ind()),
                          "Slice size for dim {0} is expected to be dynamic for correction", dim);

        auto dimOffset = offsets[dim.ind()];
        auto blockArgOffset = mlir::dyn_cast<mlir::BlockArgument>(mlir::cast<mlir::Value>(dimOffset));

        auto affineMin = mlir::cast<mlir::Value>(sizes[dim.ind()]).getDefiningOp<mlir::affine::AffineMinOp>();

        if (blockArgOffset == nullptr || affineMin == nullptr) {
            continue;
        }

        if (auto loopOp = mlir::dyn_cast<mlir::LoopLikeOpInterface>(blockArgOffset.getOwner()->getParentOp())) {
            VPUX_THROW_WHEN(!loopOp.getLoopInductionVars().has_value() || loopOp.getLoopInductionVars()->size() != 1,
                            "The loop {0} has incorrect induction varriables", loopOp);
            VPUX_THROW_WHEN(!loopOp.getLoopSteps().has_value() || loopOp.getLoopSteps()->size() != 1,
                            "The loop {0} has incorrect steps", loopOp);

            VPUX_THROW_WHEN(loopOp.getLoopRegions().empty(), "The loop {0} has no regions", loopOp);

            auto inductionVar = loopOp.getLoopInductionVars()->front();

            if (inductionVar != blockArgOffset) {
                continue;
            }

            auto loopStep = mlir::cast<mlir::Value>(loopOp.getLoopSteps()->front());

            builder.setInsertionPointToStart(&loopOp.getLoopRegions().front()->front());

            auto mainStepBound = builder.create<mlir::arith::ConstantIndexOp>(loopOp.getLoc(), data.first);
            auto remainderStep = builder.create<mlir::arith::ConstantIndexOp>(loopOp.getLoc(), data.second);
            auto isNotRemainder = builder.create<mlir::arith::CmpIOp>(loopOp.getLoc(), mlir::arith::CmpIPredicate::ult,
                                                                      inductionVar, mainStepBound);
            auto ifOp = builder.create<mlir::scf::IfOp>(
                    loopOp.getLoc(), isNotRemainder,
                    /*thenBuilder=*/
                    [&](mlir::OpBuilder& thenBuilder, mlir::Location thenLoc) {
                        thenBuilder.create<mlir::scf::YieldOp>(thenLoc, mlir::ValueRange{inductionVar, loopStep});
                    },
                    /*elseBuilder=*/
                    [&](mlir::OpBuilder& elseBuilder, mlir::Location elseLoc) {
                        mlir::AffineExpr d0, s0, s1, s2;
                        bindDims(builder.getContext(), d0);
                        bindSymbols(elseBuilder.getContext(), s0, s1, s2);
                        mlir::AffineExpr offsetExpr = ((d0 - s0).floorDiv(s1)) * s2 + s0;

                        auto affineMap = mlir::AffineMap::get(1, 3, {offsetExpr}, elseBuilder.getContext());

                        auto newOffset = mlir::affine::makeComposedAffineApply(
                                elseBuilder, appendLoc(elseLoc, "adjusted_offset"), affineMap,
                                {inductionVar, mainStepBound.getResult(), loopStep, remainderStep.getResult()});
                        elseBuilder.create<mlir::scf::YieldOp>(elseLoc, mlir::ValueRange{newOffset, remainderStep});
                    });
            inductionVar.replaceUsesWithIf(ifOp.getResult(0), [&](mlir::OpOperand& opOperand) {
                if (ifOp->getBlock() != opOperand.getOwner()->getBlock()) {
                    return opOperand.getOwner()->getParentOfType<mlir::scf::IfOp>() != ifOp;
                }
                return ifOp->isBeforeInBlock(opOperand.getOwner());
            });
            builder.replaceOp(affineMin, ifOp.getResult(1));
        }
    }
}

/*
  Get tile size for static shape operations based on strategy
  If size of operations > 1 it means that tile size is computed for VF
  where specifics of every operation should be taken into consideration
  The list of operation doesn't guarantee the order of operations so that lastOperation might be specify separately
  For each tile dimension the maximum size from tiles from fillDividedTiles is taken
*/
SmallVector<mlir::OpFoldResult> vpux::VPU::staticTileSizeComputation(
        mlir::OpBuilder& builder, ArrayRef<mlir::Operation*> operations, mlir::Operation* lastOperation,
        ShapeRef strategy, ShapeRef outputShape, std::unordered_map<Dim, std::pair<int64_t, int64_t>>& remainders) {
    if (operations.empty()) {
        return {};
    }

    if (lastOperation == nullptr) {
        lastOperation = operations.back();
    } else if (!llvm::is_contained(operations, lastOperation)) {
        return {};
    }

    const auto dynOperationAlignment = [&](mlir::Operation* op) {
        return op == lastOperation;
    };

    const auto tiles = fillDividedTiles(operations, strategy, outputShape, dynOperationAlignment);

    if (mlir::failed(tiles) || tiles.value().empty()) {
        return {};
    }

    auto tilingDims = getSCFTilingOrderedDims(lastOperation, strategy);

    if (tilingDims.empty()) {
        lastOperation->removeAttr(tilingStrategy);
        return {};
    }
    std::unordered_map<Dim, int64_t> sizes;

    // if we have uneven distribution for tensor's size among tiles,
    // take the first value and remainder will be adjusted after tiling
    auto& firstTile = tiles.value().front();
    for (auto dim : tilingDims) {
        sizes[dim] = firstTile.shape[dim];
    }

    for (auto dim : tilingDims) {
        auto tileRemainderNumber = 0;
        auto offsetTile = 0;
        auto remainderSize = 0;
        for (auto tile : tiles.value() | reversed) {
            if (tile.shape[dim] == sizes[dim]) {
                break;
            }

            if (offsetTile == 0 || offsetTile != tile.offsets[dim]) {
                ++tileRemainderNumber;
            }

            remainderSize = tile.shape[dim];
            offsetTile = tile.offsets[dim];
        }
        if (tileRemainderNumber > 1 && offsetTile != 0) {
            remainders[dim] = std::make_pair(offsetTile, remainderSize);
        }
    }

    SmallVector<mlir::OpFoldResult> tileSizes;
    tileSizes.reserve(tilingDims.size());

    const auto tileSizeCondition = [&](auto index) -> mlir::OpFoldResult {
        return builder.getIndexAttr(sizes[tilingDims[index]]);
    };

    llvm::transform(llvm::seq<size_t>(0, tilingDims.size()), std::back_inserter(tileSizes), tileSizeCondition);

    return tileSizes;
}

/*
  Get tile size for dynamic shape operations based on strategy
  If size of operations > 1 it means that tile size is computed for VF
  where specifics of every operation should be taken into consideration
  If type of last operation is BoundedTensorType then bounds are used for static tile size computation
  The list of operation doesn't guarantee the order of operations so that lastOperation might be specify separately
  If not, tileSize is computed based on formula (shape value) / divisor + alignment - 1) / alignment
  where divisor is taken from strategy and alignment is taken from operation attribute if exists or set to 1
*/
SmallVector<mlir::OpFoldResult> vpux::VPU::dynamicTileSizeComputation(mlir::OpBuilder& builder,
                                                                      ArrayRef<mlir::Operation*> operations,
                                                                      mlir::Operation* lastOperation, ShapeRef strategy,
                                                                      bool useBoundedType) {
    if (operations.empty()) {
        return {};
    }

    if (lastOperation == nullptr) {
        lastOperation = operations.back();
    } else if (!llvm::is_contained(operations, lastOperation)) {
        return {};
    }

    auto outputType = mlir::cast<mlir::ShapedType>(lastOperation->getResult(0).getType());

    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(outputType);
        boundedType != nullptr && useBoundedType) {
        auto bounds = to_small_vector(boundedType.getBounds());
        std::unordered_map<Dim, std::pair<int64_t, int64_t>> emptyRemainders;
        return staticTileSizeComputation(builder, operations, lastOperation, strategy, ShapeRef(bounds),
                                         emptyRemainders);
    }

    auto outputShape = outputType.getShape();

    SmallVector<mlir::OpFoldResult> tileSizes;

    auto tilingDims = getSCFTilingOrderedDims(lastOperation, strategy);
    tileSizes.reserve(tilingDims.size());

    for (auto tileDim : tilingDims) {
        VPUX_THROW_WHEN(!outputType.isDynamicDim(tileDim.ind()), "Tiled axis {0} must be dynamic", tileDim);

        auto loc = lastOperation->getLoc();

        auto shapeValue = getDimValue(builder, lastOperation, tileDim.ind());

        const auto alignments = vpux::getAlignment(lastOperation, strategy, ShapeRef(outputShape));
        const auto divisor = strategy[tileDim];
        const auto alignment = alignments[tileDim.ind()];

        mlir::OpFoldResult tileSize;
        mlir::AffineExpr d0;
        bindDims(builder.getContext(), d0);
        auto tileSizeMap = mlir::AffineMap::get(1, 0, {(d0.ceilDiv(divisor) + alignment - 1).floorDiv(alignment)},
                                                builder.getContext());
        tileSize = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "tileSize"), tileSizeMap,
                                                               {shapeValue});
        tileSizes.emplace_back(tileSize);
    }

    return tileSizes;
}

mlir::LogicalResult vpux::VPU::applySCFTiling(mlir::Operation* operation, mlir::RewriterBase& builder) {
    if (!operation->hasAttr(tilingStrategy)) {
        return mlir::failure();
    }
    const auto strategy =
            Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));

    mlir::scf::SCFTilingOptions tilingOptions;
    std::unordered_map<Dim, std::pair<int64_t, int64_t>> remainders;

    // Runs synchronously within a single applySCFTiling call; cachedTileSizes is only reused here.
    // TODO: Revisit if tiling is ever invoked concurrently.
    SmallVector<mlir::OpFoldResult> cachedTileSizes;
    const auto tileSizeComputationFnc = [&](mlir::OpBuilder&, mlir::Operation*) {
        if (getShape(operation->getResult(0)).isDynamic()) {
            return dynamicTileSizeComputation(builder, {operation}, nullptr, strategy);
        }
        return staticTileSizeComputation(builder, {operation}, nullptr, strategy, getShape(operation->getResult(0)),
                                         remainders);
    };

    cachedTileSizes = tileSizeComputationFnc(builder, operation);
    if (cachedTileSizes.empty() && VPU::hasDynamicDimAlignment(operation)) {
        VPU::removeDynamicDimAlignment(operation);
        cachedTileSizes = tileSizeComputationFnc(builder, operation);
    }

    if (cachedTileSizes.empty()) {
        return mlir::failure();
    }

    // Capture by value; no shared state. Tile sizes are computed eagerly—no speedup from lazy evaluation inside
    // tileUsingSCF.
    tilingOptions.setTileSizes(cachedTileSizes);

    auto tilingResult = mlir::scf::tileUsingSCF(builder, mlir::cast<mlir::TilingInterface>(operation), tilingOptions);
    if (mlir::failed(tilingResult) || tilingResult->loops.empty()) {
        return mlir::failure();
    }

    // E-162999 rewrite to update order attribute for output types more elegantly
    // tileUsingSCF drops the output order in the ForOp and terminator. This adds it back.
    auto outputType = operation->getResult(0).getType();
    llvm::for_each(tilingResult->loops, [&](mlir::LoopLikeOpInterface loop) {
        auto forOp = mlir::cast<mlir::scf::ForOp>(loop.getOperation());
        forOp.getResult(0).setType(outputType);

        auto* terminator = forOp.getBody()->getTerminator();
        if (terminator == nullptr) {
            return;
        }
        llvm::for_each(terminator->getOperands(), [&](mlir::Value operand) {
            operand.setType(outputType);

            if (auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operand.getDefiningOp())) {
                insertSlice.getDestMutable().get().setType(outputType);
                if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertSlice.getDest())) {
                    auto argIndex = blockArg.getArgNumber() - forOp.getNumInductionVars();
                    forOp.getInitArgs()[argIndex].setType(outputType);
                }
                // insert_slice is in innermost loop
                correctOffsetAndSizeByRemainder(builder, insertSlice, remainders);
            } else {
                // outer loop has no insertSlice op, modify init args by setting order to the last one
                forOp.getInitArgs().back().setType(outputType);
            }
        });
    });

    builder.replaceOp(operation, tilingResult->replacements);

    return mlir::success();
}
// Listener to track which original producers have been tiled.
// Uses operation name + location as fingerprint to uniquely identify operations.
// Note: We only track that an original op was tiled (in a set), not store pointer to tiled op,
// because tiled op pointers can become invalid during subsequent loop restructuring.
class TiledOpsTrackingListener : public mlir::RewriterBase::ForwardingListener {
public:
    explicit TiledOpsTrackingListener(llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement>& skipConnectionMap,
                                      mlir::OpBuilder::Listener* previousListener = nullptr,
                                      Logger& log = Logger::global())
            : ForwardingListener(previousListener), skipConnectionMap(skipConnectionMap), log(log) {
    }

    void expectProducerFusion(mlir::Operation* originalProducer) {
        pendingProducers.emplace_back(originalProducer, originalProducer->getName(), originalProducer->getLoc());
        log.debug("Expecting producer {0} to be tiled for fusion", originalProducer->getName());
    }

    void notifyOperationInserted(mlir::Operation* op, mlir::OpBuilder::InsertPoint insertPoint) override {
        if (!mlir::isa<mlir::TilingInterface>(op) || op->getNumResults() == 0) {
            ForwardingListener::notifyOperationInserted(op, insertPoint);
            return;
        }
        // Track newly inserted TilingInterface operations that match pending producers
        // Find the first pending producer with matching operation name and location (FIFO order)
        auto it = std::find_if(pendingProducers.begin(), pendingProducers.end(),
                               [&](const std::tuple<mlir::Operation*, mlir::OperationName, mlir::Location>& entry) {
                                   return std::get<1>(entry) == op->getName() && std::get<2>(entry) == op->getLoc();
                               });
        if (it == pendingProducers.end()) {
            ForwardingListener::notifyOperationInserted(op, insertPoint);
            return;
        }
        // Check whether this newly inserted tiled op corresponds to the preselected
        // 'biggest user' branch of any skip-connection source.
        // If yes, mark biggestUserTiled=true so fusion control can later allow fusing
        // the shared skip-source producer only after the largest branch is materialized.
        auto skipConnectionIter = llvm::find_if(skipConnectionMap, [&](const auto& entry) {
            return entry.second.biggestUserOp == std::get<0>(*it);
        });
        if (skipConnectionIter != skipConnectionMap.end()) {
            skipConnectionIter->second.biggestUserTiled = true;
            log.debug(" Biggest user of skip connection op {0} has been tiled: {1}",
                      std::get<0>(*skipConnectionIter)->getName(), op->getName());
            ForwardingListener::notifyOperationInserted(op, insertPoint);
            return;
        }
        // Track fusion progress for skip-connection source producers:
        // store the currently materialized tiled result (`tiledValue`) of the original producer.
        // This value is later used by deferred replacement logic to rewire ExtractSlice users
        // from smaller branches to slices derived from the largest-branch tiled tensor.
        // Each time the skip-source operation is updated/tiled/fused, this callback is triggered
        // again because we intentionally keep that producer in `pendingProducers` (do not erase it
        // in this path). This lets us keep `tiledValue` synchronized with the latest materialized
        // result.
        if (skipConnectionMap.contains(std::get<0>(*it))) {
            skipConnectionMap.find(std::get<0>(*it))->getSecond().tiledValue = op->getResult(0);
            log.debug("Producer of skip connection has been tiled: {0}", op->getName());
        } else {
            // Clear pendingProducers only when the tiled operation is not a skip-connection producer.
            // We need to keep tracking operations that originate skip connections whenever they are
            // tiled or updated.
            // scf::tileConsumerAndFuseProducersUsingSCF may update this operation after tiling/fusion,
            // so we keep tracking it until the transformation finishes to keep tiledValue up to date.
            pendingProducers.erase(it);
        }

        ForwardingListener::notifyOperationInserted(op, insertPoint);
    }

private:
    // Queue of (original operation, operation name, location) tuples - maintains insertion order
    SmallVector<std::tuple<mlir::Operation*, mlir::OperationName, mlir::Location>, 8> pendingProducers;
    llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement>& skipConnectionMap;
    Logger log;
};

llvm::SetVector<mlir::Operation*> collectTiledAndFusedOps(mlir::Operation* op) {
    SmallVector<mlir::Operation*> worklist;
    llvm::SetVector<mlir::Operation*> producers;
    worklist.push_back(op);
    producers.insert(op);
    while (!worklist.empty()) {
        mlir::Operation* current = worklist.pop_back_val();
        for (mlir::OpOperand& operand : current->getOpOperands()) {
            mlir::Operation* producer = operand.get().getDefiningOp();
            const auto checkProducersUsers = [&](auto* user) {
                return !producers.contains(user);
            };
            if (!mlir::isa_and_nonnull<mlir::TilingInterface>(producer) || producers.contains(producer) ||
                !vpux::VPU::checkFusion(operand, producer->getOpResult(0), producers) ||
                llvm::any_of(producer->getUsers(), checkProducersUsers)) {
                continue;
            }
            worklist.push_back(producer);
            producers.insert(producer);
        }
    }
    return producers;
}

VPU::VF::v2::VFSplit getVFSplit(vpux::NDTypeInterface outputType, mlir::Operation* outputOperation,
                                DimArrRef allowedDims, VPU::VF::v2::VFConfig& config) {
    auto shapeType = mlir::cast<mlir::ShapedType>(outputType);
    if (!shapeType.hasStaticShape()) {
        VPU::VF::v2::VFSplit vfSplit;
        auto countDynDims = shapeType.getNumDynamicDims();
        // allowedDims to memoryOrder
        auto outputOrder = outputType.getDimsOrder();
        auto allowedDimsInMemoryOrder = allowedDims.vec();
        llvm::sort(allowedDimsInMemoryOrder, [&](const Dim& d1, const Dim& d2) {
            return outputOrder.dimPos(d1) < outputOrder.dimPos(d2);
        });
        const auto hasDynAlignment = VPU::hasDynamicDimAlignment(outputOperation);
        auto boundedShape = getBoundedShape(outputType);
        std::optional<SmallVector<int64_t>> alignmentValues;
        auto innerMostDynamicDim = getInnermostDynamicDim(outputType.getShape(), outputType.getDimsOrder());
        if (hasDynAlignment) {
            alignmentValues = getAlignment(outputOperation, {}, boundedShape, true);
        }

        for (auto dim : allowedDimsInMemoryOrder) {
            if (!shapeType.isDynamicDim(dim.ind())) {
                continue;
            }

            VPUX_THROW_WHEN(dim == Dims4D::Act::C, "Dynamic channels are not supported");
            if (hasDynAlignment && alignmentValues.has_value() && innerMostDynamicDim.has_value()) {
                if (dim == innerMostDynamicDim.value() && countDynDims > 1) {
                    vfSplit[dim] = divUp(boundedShape[dim], alignmentValues.value()[dim.ind()]);
                } else {
                    vfSplit[dim] = std::nullopt;
                }
                continue;
            }
            if (countDynDims == 1) {
                vfSplit[dim] = std::nullopt;
                break;
            }

            vfSplit[dim] = getTilingLimit(dim, config, true);
            --countDynDims;
        }
        return vfSplit;
    }

    VPU::SiblingOpsAnalysis siblingAnalisys(outputOperation);
    auto outputShape = outputType.getShape().toValues();
    if (auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(outputOperation)) {
        if (clusterOp.getMultiClusterStrategy().has_value()) {
            outputType = clusterOp.getDistributedTypeForOpResult(
                    outputOperation->getResult(0), clusterOp.getMultiClusterStrategy().value(), siblingAnalisys, false);

            auto distribution = VPU::DistributionInfo::getClassFromAttr(
                    mlir::cast<VPU::DistributedTensorType>(outputType).getDistribution());
            if (distribution.getMemoryShapes().empty()) {
                auto optMemoryShapes =
                        VPU::getPerClusterMemoryShapes(outputShape, distribution, outputType.getElementType());
                if (optMemoryShapes.has_value()) {
                    outputShape = Shape(optMemoryShapes.value().front());
                }
            } else {
                outputShape = Shape(distribution.getMemoryShapes().front());
            }
        }
    }

    const auto compareDims = [&](auto dimLeft, auto dimRight) {
        return outputShape[dimLeft] < outputShape[dimRight];
    };
    auto maxDim = std::max_element(allowedDims.begin(), allowedDims.end(), compareDims);

    if (maxDim == allowedDims.end()) {
        return {};
    }

    return {{*maxDim, std::nullopt}};
}

namespace {
VPU::VF::v2::VFCase computeVFSCFCase(vpux::NDTypeInterface outputType, mlir::Operation* lastOp, DimArrRef allowedDims,
                                     VPU::VF::v2::VFConfig& config, mlir::Operation* rootOp, Logger log) {
    VPU::VF::v2::VFCase emptyVFCase(config, {});
    auto vfSplit = getVFSplit(outputType, lastOp, allowedDims, config);
    if (vfSplit.empty()) {
        return emptyVFCase;
    }

    auto optDim = VPU::VF::v2::getNonTiledDimForVFOptimization(vfSplit);
    if (!optDim.has_value()) {
        return emptyVFCase;
    }

    const auto getMinTiles = [&](auto dim, const VPU::VF::v2::VFSplit& split) {
        if ((outputType.getShape().isDynamic() && dim == optDim.value()) || split.size() > 1) {
            return VPU::MIN_REQUIRED_TILES;
        }

        const auto getDimValue = [&dim](auto* oper) -> int64_t {
            return oper->hasAttr(tilingStrategy) ? Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(
                                                           oper->getAttr(tilingStrategy))))[dim]
                                                 : 1;
        };

        std::set<int64_t> minTilesSet;
        llvm::copy(config.getVFOperations() | transformed(getDimValue), std::inserter(minTilesSet, minTilesSet.end()));
        return *minTilesSet.rbegin();
    };

    const auto getMaxTiles = [&](auto dim, const VPU::VF::v2::VFSplit& split) -> int64_t {
        auto maxTiles = getTilingLimit(dim, config);
        if (!VPU::hasDynamicDimAlignment(rootOp) && split.size() > 1) {
            auto otherDimSum = VPU::VF::v2::getVFTilesLen(split);
            maxTiles = divUp(maxTiles, otherDimSum);

            if (outputType.getShape().isDynamic()) {
                maxTiles = std::max(maxTiles, VPU::MINIMUM_LENGTH_TILING);
            }
        }
        return maxTiles;
    };

    return VPU::VF::v2::getVFCaseWithTiling(config, optDim.value(), vfSplit, getMinTiles, getMaxTiles, log,
                                            VPU::VF::v2::getSchedulingScenarios(config, log));
}
}  // namespace

SmallVector<mlir::Operation*> vpux::VPU::applySCFVerticalFusion(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                                Logger log) {
    if (!operation->hasAttr(tilingStrategy)) {
        return {};
    }

    auto tilingInterfaceOp = mlir::cast<mlir::TilingInterface>(operation);

    const auto strategy = operation->getAttr(tilingStrategy);
    mlir::scf::SCFTilingOptions tilingOptions;

    // calculate tile size based on VF restrictions
    auto allOpsToFuse = collectTiledAndFusedOps(operation);

    if (allOpsToFuse.size() == 1) {
        return {};
    }

    VF::v2::VFConfig config(allOpsToFuse);
    DimArr allowedDims = getAllowedDims(allOpsToFuse.getArrayRef(), log);
    if (allowedDims.empty()) {
        return {};
    }

    auto outputs = config.getOutputs();
    if (outputs.empty()) {
        return {};
    }

    auto* lastOp = outputs.back();
    auto outputType = mlir::cast<vpux::NDTypeInterface>(lastOp->getResult(0).getType());

    VPU::VF::v2::VFCase bestVFCase = computeVFSCFCase(outputType, lastOp, allowedDims, config, operation, log);

    const auto& tilingStorage = bestVFCase.getTilingStorage();

    // Build skip connection map: operations with multiple uses and user op which require biggest tile
    llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement> skipConnectionMap =
            analyzeSkipConnectionsForTiling(allOpsToFuse, tilingStorage, log);

    if (VPU::hasDynamicDimAlignment(operation) && !bestVFCase.isInitialized()) {
        // We are retrying VF search with disabled dynamic alignment
        VPU::removeDynamicDimAlignment(operation);
        bestVFCase = computeVFSCFCase(outputType, lastOp, allowedDims, config, operation, log);
    }

    if (!bestVFCase.isInitialized()) {
        return {};
    }

    operation->setAttr(tilingStrategy, bestVFCase.getTiling());
    std::unordered_map<Dim, std::pair<int64_t, int64_t>> remainders;

    // calculate tile size for VF:
    // 1. allOpsToFuse contains operations to build VF
    // 2. get all allowed dimensions for these operations to tile
    // 3. choose the dimension which tiles the largest dimension
    // 4. get optimal tiling number for vertical fusion
    // 5. calculate tile size based on computed tiling number
    const auto vfTileSizeComputationFn = [&](mlir::OpBuilder& builder,
                                             mlir::Operation* operation) -> SmallVector<mlir::OpFoldResult> {
        auto strategy = Shape(parseIntArrayAttr<int64_t>(bestVFCase.getTiling()));

        if (outputType.getShape().isStatic()) {
            return staticTileSizeComputation(builder, allOpsToFuse.getArrayRef(), operation, strategy,
                                             getShape(operation->getResult(0)), remainders);
        }

        return dynamicTileSizeComputation(builder, allOpsToFuse.getArrayRef(), operation, strategy);
    };

    tilingOptions.setTileSizeComputationFunction(vfTileSizeComputationFn);

    mlir::scf::SCFTileAndFuseOptions tilingAndFuseOptions;
    tilingAndFuseOptions.setTilingOptions(std::move(tilingOptions));

    SmallVector<std::pair<mlir::tensor::ExtractSliceOp, mlir::Value>> pendingSliceReplacements;
    // check if VF has loop to substitute slice with existing operation
    bool hasSkipConnection = false;

    // Save the previous listener and create our tracking listener
    auto* previousListener = builder.getListener();
    TiledOpsTrackingListener listener(skipConnectionMap, previousListener, log);
    builder.setListener(&listener);

    mlir::scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
            [&](mlir::tensor::ExtractSliceOp sliceOp, mlir::OpResult originalProducer,
                bool) -> std::optional<mlir::scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (!allOpsToFuse.contains(originalProducer.getOwner())) {
            // Return an empty optional to signal "do not fuse".
            return std::nullopt;
        }
        auto skipConnectionIter = skipConnectionMap.find(originalProducer.getOwner());
        if (skipConnectionIter == skipConnectionMap.end()) {
            // if skip connection op is not involved - fuse as usual
            originalProducer.getOwner()->setAttr(tilingStrategy, bestVFCase.getTiling());
            // Notify listener that this producer is about to be fused
            listener.expectProducerFusion(originalProducer.getOwner());
            // Return a result to signal "fuse this op".
            return mlir::scf::SCFTileAndFuseOptions::ControlFnResult{};
        }
        log.debug("Attempting to fuse producer of skip connection: {0} at loc: {1}",
                  originalProducer.getOwner()->getName(), originalProducer.getOwner()->getLoc());
        auto& deferredReplacement = skipConnectionIter->getSecond();

        // Check if Operation which originates skip connection has been tiled already
        if (deferredReplacement.tiledValue != nullptr) {
            log.debug("Operation which originates skip connection has been tiled already. Save current sliceOp "
                      "for future replacement.\n");
            deferredReplacement.relatedExtractSlices.insert(sliceOp);
            // Return an empty optional to signal "do not fuse".
            return std::nullopt;
        }

        if (!deferredReplacement.biggestUserTiled && !deferredReplacement.allUsersWithTheSameTileSize) {
            log.debug("Biggest user not tiled yet, cannot fuse producer. Skipping fusion for this producer");
            deferredReplacement.relatedExtractSlices.insert(sliceOp);
            // Return an empty optional to signal "do not fuse".
            return std::nullopt;
        }

        // If the skip-source op has not been tiled yet, allow fusion only after the largest user branch was
        // tiled (or when all branches have the same tile size).
        // Assumption: tile+fuse walks a single branch bottom-up. It tiles users first and then continues
        // upward along the same branch to the skip-source producer.
        // With this assumption, once the largest branch is tiled we proceed and expect the next step to tile
        // and fuse the skip-source from that same branch, not from a different branch.
        log.debug("Biggest User tiled (or all users have the same tile size), allowing fusion");
        // Treat current sliceOp as the biggest-branch slice and remember it for future replacements.
        deferredReplacement.biggestTileExtractSlice = sliceOp;

        originalProducer.getOwner()->setAttr(tilingStrategy, bestVFCase.getTiling());
        listener.expectProducerFusion(originalProducer.getOwner());
        hasSkipConnection = true;
        // Return a result to signal "fuse this op".
        return mlir::scf::SCFTileAndFuseOptions::ControlFnResult{};
    };
    tilingAndFuseOptions.setFusionControlFn(std::move(controlFn));
    builder.setInsertionPoint(operation);

    auto tiledResults =
            mlir::scf::tileConsumerAndFuseProducersUsingSCF(builder, tilingInterfaceOp, tilingAndFuseOptions);

    if (mlir::failed(tiledResults) || tiledResults->replacements.empty() || tiledResults->loops.empty() ||
        tiledResults->fusedProducers.empty()) {
        operation->setAttr(tilingStrategy, strategy);
        builder.setListener(previousListener);
        return {};
    }

    applyDeferredSliceReplacements(builder, skipConnectionMap, log);

    // propagate result type with order and bounds attributes to operations
    // created in SCF functions.
    for (auto result : operation->getResults()) {
        tiledResults->replacements[result].setType(result.getType());
    }

    // E-162999 rewrite to update order attribute for output types more elegantly
    llvm::for_each(tiledResults->loops, [&](mlir::LoopLikeOpInterface loopOperation) {
        auto loop = mlir::cast<mlir::scf::ForOp>(loopOperation);

        auto* terminator = loop.getBody()->getTerminator();
        if (terminator == nullptr) {
            return;
        }
        llvm::for_each(terminator->getOperands(), [&](mlir::Value operand) {
            operand.setType(loop.getResult(0).getType());

            auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operand.getDefiningOp());
            if (insertSlice == nullptr) {
                // outer loop has no insertSlice op, modify init args by setting order to the last one
                loop.getInitArgs().back().setType(operand.getType());
                return;
            }
            // insert_slice is in innermost loop
            correctOffsetAndSizeByRemainder(builder, insertSlice, remainders);
            insertSlice.getDestMutable().get().setType(loop.getResult(0).getType());
            if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertSlice.getDest())) {
                auto argIndex = blockArg.getArgNumber() - loop.getNumInductionVars();
                loop.getInitArgs()[argIndex].setType(operand.getType());
            }
            if (hasSkipConnection) {
                // reorder operations in the loop body to ensure that operation in the loop
                // is in order after resolving circle dependencies
                vpux::VPU::reorderOperations(
                        to_small_vector(loop.getBody()->without_terminator() | transformed([](mlir::Operation& op) {
                                            return &op;
                                        })));
            }
        });
    });

    for (mlir::OpResult res : operation->getResults()) {
        if (auto replacement = tiledResults->replacements.lookup(res)) {
            builder.replaceAllUsesWith(res, replacement);
        }
    }

    if (operation->use_empty()) {
        builder.eraseOp(operation);
    }

    builder.setListener(previousListener);

    return to_small_vector(tiledResults->fusedProducers);
}
