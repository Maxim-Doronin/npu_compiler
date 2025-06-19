//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"

namespace vpux::VPU {

/** @brief Information about a tile.

    The structure incapsulates data of offsets, shape and tile axis
    for a tensor represented as mlir::OpFoldResult
*/

using SCFShape = SmallVector<mlir::OpFoldResult>;
using SCFShapeRef = ArrayRef<mlir::OpFoldResult>;

struct SCFTileInfo {
    SCFShape shape;
    SCFShape offsets;
    SCFShape axis;

    SCFTileInfo() = delete;

    explicit SCFTileInfo(SCFShapeRef shape, SCFShapeRef offsets, SCFShapeRef axis)
            : shape(shape), offsets(offsets), axis(axis) {
    }

    explicit SCFTileInfo(ArrayRef<int64_t> shapeInt, mlir::OpBuilder& builder)
            : shape(mlir::getAsIndexOpFoldResult(builder.getContext(), shapeInt)),
              offsets(SCFShape(shapeInt.size(), builder.getIndexAttr(0))),
              axis(SCFShape(shapeInt.size(), builder.getIndexAttr(1))) {
    }

    explicit SCFTileInfo(SCFShapeRef shape, mlir::OpBuilder& builder)
            : shape(shape),
              offsets(SCFShape(shape.size(), builder.getIndexAttr(0))),
              axis(SCFShape(shape.size(), builder.getIndexAttr(1))) {
    }
};

using OpTilingOperandsFunc = std::function<void(SmallVector<SCFTileInfo>&)>;
using OpGeneratorFunc = std::function<mlir::Operation*()>;
using SCFTilingInfo = SmallVector<SCFTileInfo>;

// @brief Dim value of input/output/weights shape
mlir::OpFoldResult getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim);

// @brief Calculates tile for weights table based on output tile
SCFTileInfo getWeightsTableSCFTile(mlir::Type origWeightsTableType, mlir::OpBuilder& builder,
                                   const SCFTileInfo& outputTile);

/** @brief Restores input tiling from output tile data

    The function calculates input shape and offset based on
    parameters and shape and offset of output tile
*/
mlir::Range solutionForOutputRange(mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile, Dim dim,
                                   const int64_t kernel, const int64_t stride,
                                   const std::pair<int64_t, int64_t>& origPadding);

/** @brief create operation with padding adjustment

    @note If operation has paddings which are not 0, they have to be corrected based on
    position of tile. Unfortunately, in OV based operations paddings have to be known integer attributes,
    they cannot be calculated or created as constant for each case. That's why there must be the structure
    with if-else which identifies how paddings are set.
*/
template <class ConcreteOp>
mlir::Operation* createTiledPaddedOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                            mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                            const SCFTileInfo& outputTile, Dim dim, SCFShapeRef origShape,
                                            mlir::Operation* origOperation, ShapeRef tiling) {
    if (dim == Dims4D::Act::C) {
        operandsGenerator(inputTiling);
        return opGenerator();
    }
    auto padInfo = toPadInfo(mlir::cast<ConcreteOp>(origOperation).getPad());
    if (!padInfo.enabled()) {
        operandsGenerator(inputTiling);
        return opGenerator();
    }

    auto numTiles = tiling[dim];
    if (numTiles == 1) {
        operandsGenerator(inputTiling);
        return opGenerator();
    }

    VPUX_THROW_WHEN(static_cast<size_t>(dim.ind()) < Dims4D::Act::numSpatialDims, "Incorrect tiling spacial dim {0}",
                    dim);

    const auto spatialDimIdx = dim.ind() - Dims4D::Act::numSpatialDims;
    auto loc = origOperation->getLoc();

    auto zeroOffset = builder.create<mlir::arith::ConstantIndexOp>(appendLoc(loc, "zero"), 0);
    auto interValue =
            mlir::getValueOrCreateConstantIndexOp(builder, appendLoc(loc, "offset"), outputTile.offsets[dim.ind()]);

    auto isFirstIndex = builder.create<mlir::arith::CmpIOp>(appendLoc(loc, "equal"), mlir::arith::CmpIPredicate::eq,
                                                            interValue, zeroOffset);

    const auto createOperation = [&](bool trimBegin, bool trimEnd) {
        auto poolingOp = mlir::cast<ConcreteOp>(opGenerator());

        std::array<int64_t, Dims4D::Act::numSpatialDims> padsBegin = {padInfo.top, padInfo.left};
        std::array<int64_t, Dims4D::Act::numSpatialDims> padsEnd = {padInfo.bottom, padInfo.right};

        if (trimBegin) {
            padsBegin[spatialDimIdx] = 0;
        }

        if (trimEnd) {
            padsEnd[spatialDimIdx] = 0;
        }

        poolingOp.setPadAttr(getPaddingAttr(builder.getContext(), padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]));
        builder.create<mlir::scf::YieldOp>(appendLoc(loc, "yield"), poolingOp.getResult());
    };

    operandsGenerator(inputTiling);

    const auto firstTileCreator = [&](mlir::OpBuilder&, mlir::Location) {
        return createOperation(/*trimBegin=*/false, /*trimEnd=*/true);
    };

    const auto lastTileCreator = [&](mlir::OpBuilder&, mlir::Location) {
        return createOperation(/*trimBegin=*/true, /*trimEnd=*/false);
    };

    const auto medianTileCreator = [&](mlir::OpBuilder& opBuilder, mlir::Location opLocation) {
        auto newInfo = inputTiling;
        auto& inputTile = newInfo[0];
        mlir::AffineExpr d0;
        bindDims(opBuilder.getContext(), d0);
        std::array<int64_t, Dims4D::Act::numSpatialDims> padsEnd = {padInfo.bottom, padInfo.right};
        auto addMap = mlir::AffineMap::get(1, 0, {d0 + padsEnd[spatialDimIdx]}, opBuilder.getContext());
        inputTile.shape[dim.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                opBuilder, appendLoc(opLocation, "paddedShape"), addMap, {inputTile.shape[dim.ind()]});
        operandsGenerator(newInfo);
        return createOperation(/*trimBegin=*/true, /*trimEnd=*/true);
    };

    const auto elseBlockCreator = [&](mlir::OpBuilder& opBuilder, mlir::Location opLocation) {
        if (numTiles == 2) {
            return createOperation(/*trimBegin=*/true, /*trimEnd=*/false);
        }

        auto maxValue = mlir::getValueOrCreateConstantIndexOp(opBuilder, appendLoc(opLocation, "maxValue"),
                                                              origShape[dim.ind()]);
        auto lastIndex = opBuilder.create<mlir::arith::SubIOp>(appendLoc(opLocation, "sub"), maxValue, interValue);

        auto isLastIndex = opBuilder.create<mlir::arith::CmpIOp>(appendLoc(opLocation, "equal"),
                                                                 mlir::arith::CmpIPredicate::eq, interValue, lastIndex);
        auto innerIfOp = opBuilder.create<mlir::scf::IfOp>(appendLoc(opLocation, "innerIf"), isLastIndex,
                                                           lastTileCreator, medianTileCreator);
        opBuilder.create<mlir::scf::YieldOp>(appendLoc(opLocation, "yield"), innerIfOp.getResult(0));
    };

    auto ifOp = builder.create<mlir::scf::IfOp>(appendLoc(loc, "outerIf"), isFirstIndex, firstTileCreator,
                                                elseBlockCreator);

    return ifOp.getOperation();
}

}  // namespace vpux::VPU
