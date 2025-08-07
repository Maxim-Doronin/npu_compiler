//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

using namespace vpux;

SmallVector<mlir::OpFoldResult> vpux::VPU::staticTileSizeComputation(mlir::OpBuilder& builder,
                                                                     mlir::Operation* operation, ShapeRef strategy,
                                                                     ShapeRef outputShape) {
    const auto tiles = fillDividedTiles(operation, strategy, outputShape);

    if (mlir::failed(tiles)) {
        return {};
    }

    auto tilingDims = getNonOneDim(strategy);
    std::unordered_map<Dim, int64_t> sizes;

    for (auto& tile : tiles.value()) {
        for (auto dim : tilingDims) {
            sizes[dim] = std::max(tile.shape[dim], sizes[dim]);
        }
    }

    SmallVector<mlir::OpFoldResult> tileSizes;
    tileSizes.reserve(tilingDims.size());

    const auto tileSizeCondition = [&](auto& sizePair) -> mlir::OpFoldResult {
        return builder.getIndexAttr(sizePair.second);
    };

    llvm::transform(sizes, std::back_inserter(tileSizes), tileSizeCondition);

    return tileSizes;
}

SmallVector<mlir::OpFoldResult> vpux::VPU::dynamicTileSizeComputation(mlir::OpBuilder& builder,
                                                                      mlir::Operation* operation, ShapeRef strategy) {
    // E-162801 extend to multi axes tiling
    auto tilingDims = getNonOneDim(strategy);
    VPUX_THROW_WHEN(tilingDims.size() != 1, "Unsupported tiling strategy for dynamic shapes");
    auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());

    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(outputType)) {
        auto bounds = to_small_vector(boundedType.getBounds());
        return staticTileSizeComputation(builder, operation, strategy, Shape(bounds));
    }

    auto outputShape = outputType.getShape();
    const auto tileDim = tilingDims.front();

    VPUX_THROW_WHEN(!outputType.isDynamicDim(tileDim.ind()), "Tiled axis {0} must be dynamic", tileDim);

    auto loc = operation->getLoc();

    auto shapeValue = getDimValue(builder, operation, tileDim.ind());

    auto optAlignment = vpux::getAlignment(operation, strategy, Shape(outputShape));
    const auto divisor = strategy[tileDim];
    const auto alignment = optAlignment.has_value() ? optAlignment.value()[tileDim.ind()] : 1;

    mlir::OpFoldResult tileSize;
    mlir::AffineExpr d0;
    bindDims(builder.getContext(), d0);
    auto tileSizeMap = mlir::AffineMap::get(1, 0, {(d0.ceilDiv(divisor) + alignment - 1).floorDiv(alignment)},
                                            builder.getContext());
    tileSize =
            mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "tileSize"), tileSizeMap, {shapeValue});

    return {tileSize};
}

mlir::LogicalResult vpux::VPU::applySCFTiling(mlir::Operation* operation, mlir::RewriterBase& builder) {
    if (!operation->hasAttr(tilingStrategy)) {
        return mlir::failure();
    }
    const auto strategy =
            Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));

    mlir::scf::SCFTilingOptions tilingOptions;

    const auto tileSizeComputationFnc = [&](mlir::OpBuilder&, mlir::Operation*) {
        if (getShape(operation->getResult(0)).isDynamic()) {
            return dynamicTileSizeComputation(builder, operation, strategy);
        }

        return staticTileSizeComputation(builder, operation, strategy, getShape(operation->getResult(0)));
    };

    tilingOptions.setTileSizeComputationFunction(tileSizeComputationFnc);

    auto tilingResult = mlir::scf::tileUsingSCF(builder, mlir::cast<mlir::TilingInterface>(operation), tilingOptions);
    if (mlir::failed(tilingResult) || tilingResult->replacements.empty() ||
        tilingResult->replacements.size() != operation->getNumResults() || tilingResult->loops.empty()) {
        return mlir::failure();
    }

    for (auto [result, loopOutput] : llvm::zip(operation->getResults(), tilingResult->replacements)) {
        loopOutput.setType(result.getType());
    }

    // E-162999 rewrite to update order attribute for output types more elegantly
    llvm::for_each(tilingResult->loops, [&](mlir::LoopLikeOpInterface loop) {
        auto forOp = mlir::cast<mlir::scf::ForOp>(loop.getOperation());

        auto* terminator = forOp.getBody()->getTerminator();
        if (terminator != nullptr) {
            llvm::for_each(terminator->getOperands(), [&](mlir::Value operand) {
                operand.setType(forOp.getResult(0).getType());

                if (auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operand.getDefiningOp())) {
                    insertSlice.getDestMutable().get().setType(forOp.getResult(0).getType());
                    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertSlice.getDest())) {
                        auto argIndex = blockArg.getArgNumber() - forOp.getNumInductionVars();
                        forOp.getInitArgs()[argIndex].setType(operand.getType());
                    }
                }
            });
        }
    });

    builder.replaceOp(operation, tilingResult->replacements);
    return mlir::success();
}
