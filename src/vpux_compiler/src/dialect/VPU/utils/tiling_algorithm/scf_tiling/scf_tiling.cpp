//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

using namespace vpux;

SmallVector<mlir::OpFoldResult> vpux::VPU::staticTileSizeComputation(mlir::OpBuilder& builder,
                                                                     mlir::Operation* operation, ShapeRef strategy) {
    auto tilingDims = getNonOneDim(strategy);

    const auto tiles = fillDividedTiles(operation, strategy, getShape(operation->getResult(0)));

    if (mlir::failed(tiles)) {
        return {};
    }

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

mlir::LogicalResult vpux::VPU::applySCFTiling(mlir::Operation* operation, mlir::RewriterBase& builder) {
    if (!operation->hasAttr(tilingStrategy)) {
        return mlir::failure();
    }
    const auto strategy = Shape(parseIntArrayAttr<int64_t>(operation->getAttr(tilingStrategy).cast<mlir::ArrayAttr>()));

    mlir::scf::SCFTilingOptions tilingOptions;
    // E-162627 support dynamic shapes in tile size calculation
    tilingOptions.setTileSizeComputationFunction(
            std::bind(vpux::VPU::staticTileSizeComputation, std::placeholders::_1, std::placeholders::_2, strategy));

    auto tilingResult =
            mlir::scf::tileUsingSCFForOp(builder, mlir::cast<mlir::TilingInterface>(operation), tilingOptions);
    if (mlir::failed(tilingResult) || tilingResult->replacements.empty() ||
        tilingResult->replacements.size() != operation->getNumResults() || tilingResult->loops.empty()) {
        return mlir::failure();
    }

    for (auto [result, loopOutput] : llvm::zip(operation->getResults(), tilingResult->replacements)) {
        loopOutput.setType(result.getType());
    }

    // E-162999 rewrite to update order attribute for output types more elegantly
    llvm::for_each(tilingResult->loops, [&](auto* operation) {
        auto loop = mlir::cast<mlir::scf::ForOp>(operation);

        auto* terminator = loop.getBody()->getTerminator();
        if (terminator != nullptr) {
            llvm::for_each(terminator->getOperands(), [&](mlir::Value operand) {
                operand.setType(loop.getResult(0).getType());

                if (auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operand.getDefiningOp())) {
                    insertSlice.getDestMutable().get().setType(loop.getResult(0).getType());
                    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertSlice.getDest())) {
                        auto argIndex = blockArg.getArgNumber() - loop.getNumInductionVars();
                        loop.getInitArgs()[argIndex].setType(operand.getType());
                    }
                }
            });
        }
    });

    builder.replaceOp(operation, tilingResult->replacements);
    return mlir::success();
}
