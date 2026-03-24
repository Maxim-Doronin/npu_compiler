//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTOPTODMAFORPERFORMANTEXECUTION
#define GEN_PASS_DEF_CONVERTOPTODMAFORPERFORMANTEXECUTION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// TileGatherElement
//

class TileGatherElement final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    TileGatherElement(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("TileGatherElement");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileGatherElement::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile*/ true, /*isIndicesTile*/ false, _log)) {
        return mlir::failure();
    }

    size_t axis = origOp.getAxisValue().value();
    const auto inputShape = getShape(origOp.getInput());
    const auto outputShape = getShape(origOp.getOutput());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto arch = config::getArch(origOp);

    Shape nTilesOnDim(outputShape.size(), 1);
    DimArr tileDimOrder;
    // Tiling the dim after axis. Gather Output shape size is different from input size, but the dim after axis will
    // keep.
    auto shapeSizeDiff = outputShape.size() - inputShape.size();
    for (size_t idx = axis + 1; idx < inputShape.size(); ++idx) {
        tileDimOrder.push_back(vpux::Dim(idx + shapeSizeDiff));
    }

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(origOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        const size_t GATHER_DMA_MAX_ELEMENT_SIZE_ARCH_BASED = VPU::getGatherDMAMaxElementSize(arch);

        for (const auto& tile : tiles.value()) {
            size_t elementSizeInBit = vpux::getElemTypeSize(outputType).count();
            auto inputTiling = origOp.backInferTileInfo(tile, _log);
            auto& inTiles = inputTiling.tiles;
            for (size_t idx = axis + 1; idx < inputShape.size(); ++idx) {
                elementSizeInBit *= inTiles.begin()->shape.raw()[idx];
            }
            if (elementSizeInBit > GATHER_DMA_MAX_ELEMENT_SIZE_ARCH_BASED * CHAR_BIT) {
                return false;
            }
        }
        return true;
    };

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;
    while (tileDimIter < tileDimOrder.end() && !isSupportedTileSize(nTilesOnDim)) {
        if (nTilesOnDim[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            dimToTile = *(++tileDimIter);
        } else {
            ++nTilesOnDim[Dim(dimToTile)];
        }
    }

    const auto tilesNew = fillDividedTiles(origOp, nTilesOnDim, outputShape);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, _log.nest());
}

//
// TileGatherIndices
//

class TileGatherIndices final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    TileGatherIndices(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("TileGatherIndices");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileGatherIndices::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile*/ false, /*isIndicesTile*/ true, _log)) {
        return mlir::failure();
    }

    const auto outputShape = getShape(origOp.getOutput());
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(origOp.getIndices().getType());
    const auto indicesShape = indicesType.getShape();
    const auto indicesRank = origOp.getIndicesRank().value_or(indicesShape.size());
    const auto arch = config::getArch(origOp);

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(origOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        const size_t DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED = VPU::getGatherDMAMaxIndicesListLength(arch);

        for (auto tile : tiles.value()) {
            const auto inputTiling = origOp.backInferTileInfo(tile, _log);
            const auto indicesTiling = inputTiling.tiles[1];
            const auto newIndicesType = indicesType.extractDenseTile(indicesTiling.offsets, indicesTiling.shape);
            const size_t numberOfIndices = newIndicesType.getNumElements();
            if (numberOfIndices <= DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED) {
                return true;
            }
        }
        return false;
    };

    int64_t axisValue = 0;

    if (origOp.getAxisValueAttr() != nullptr) {
        axisValue = mlir::cast<mlir::IntegerAttr>(origOp.getAxisValueAttr()).getValue().getSExtValue();
    }
    if (origOp.getAxis() != nullptr) {
        auto axisConst = origOp.getAxis().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_UNLESS(axisConst != nullptr, "Only constant input is supported for axis");
        VPUX_THROW_UNLESS(axisConst.getContentAttr().isSplat(), "Axis value must be a scalar");
        const auto axisContent = axisConst.getContent();
        axisValue = axisContent.getSplatValue<int64_t>();
    }

    int64_t batchDims = 0;
    if (origOp.getBatchDimsAttr() != nullptr) {
        batchDims = mlir::cast<mlir::IntegerAttr>(origOp.getBatchDimsAttr()).getValue().getSExtValue();
    }

    const auto dimToTile = axisValue + indicesRank - batchDims - 1;
    while (!isSupportedTileSize(nTilesOnDim)) {
        if (nTilesOnDim[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            return mlir::failure();
        }
        ++nTilesOnDim[Dim(dimToTile)];
    }

    const auto tilesNew = fillDividedTiles(origOp, nTilesOnDim, outputShape);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, _log.nest());
}

//
// MoveToDMAGather
//

class MoveToDMAGather final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    MoveToDMAGather(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("MoveToDMAGather");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// GatherDMA indices only support positive values
// - If the indices is constant, iterate through the values and convert any negatives to positives
// - If the indices is dynamic, TODO: E#149660
mlir::Value handleNegativeIndices(mlir::Value indices, ShapeRef dataShape, const Dim axis,
                                  mlir::PatternRewriter& rewriter) {
    if (auto indicesCst = mlir::dyn_cast_or_null<Const::DeclareOp>(indices.getDefiningOp())) {
        const auto indicesContent = indicesCst.getContent();
        auto indicesVals = to_small_vector(indicesContent.getValues<int64_t>());
        auto firstNegativeIt = std::find_if(indicesVals.begin(), indicesVals.end(), [](int64_t val) {
            return val < 0;
        });

        if (firstNegativeIt != indicesVals.end()) {
            for (auto it = firstNegativeIt; it != indicesVals.end(); ++it) {
                if (*it < 0) {
                    *it += dataShape[axis];
                }
            }

            auto indicesType = mlir::cast<NDTypeInterface>(indicesCst.getOutput().getType());
            auto indicesStorageType = mlir::cast<mlir::RankedTensorType>(
                    indicesType.changeElemType(mlir::IntegerType::get(indicesCst.getContext(), 64)));
            auto indicesStorageAttr = Const::createConstContent(indicesStorageType, ArrayRef(indicesVals));

            return rewriter
                    .create<Const::DeclareOp>(indicesCst.getLoc(), indicesStorageType,
                                              Const::ContentAttr::get(indicesStorageAttr))
                    .getOutput();
        }
    }
    return indices;
}

// Verify if adaption passes tiled Gather op on the dim after gather axis
bool hasTilingDoneToBenefitAbsAddressing(VPU::GatherOp origOp) {
    const auto gatherAxis = origOp.getAxisValue().value();
    if (auto concatOp = mlir::dyn_cast_or_null<VPU::ConcatOp>(*origOp->getUsers().begin())) {
        auto concatAxes = VPU::getConcatAxes(concatOp);

        // Check tiling over 1 axis
        if (concatAxes.size() != 1) {
            return false;
        }
        if (*concatAxes.begin() != gatherAxis + 1) {
            return false;
        }
    }
    return true;
}

// Verify if multiclustering is needed and can be done on dim after gather axis
bool canHaveMulticlusteringToBenefitAbsAddressing(VPU::GatherDMAOp origOp) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    const auto gatherAxis = origOp.getAxisValue().value();

    size_t numTile = config::getNumOfTiles(origOp);
    if (gatherAxis == Dims4D::Act::C.ind() &&
        clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight, numTile)) {
        return true;
    }
    if (gatherAxis == Dims4D::Act::H.ind() &&
        clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverWidth, numTile)) {
        return true;
    }

    return false;
}

mlir::LogicalResult MoveToDMAGather::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp.getLoc());

    auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    auto axis = Dim(origOp.getAxisValue().value());

    auto indices = handleNegativeIndices(origOp.getIndices(), inputType.getShape(), axis, rewriter);

    auto reshapeOperand = [&](mlir::Value operand, ShapeRef newShape, const mlir::Location& location) {
        auto newShapeAttr = getIntArrayAttr(operand.getContext(), newShape);
        return rewriter.createOrFold<VPU::ReshapeOp>(location, operand, newShapeAttr);
    };

    // Ensure Indices tensor has the same rank as the Input tensor for GatherDMA
    //  - Fuse Indices into one dimension and align it with the axis dimension of Input
    //  - Fill other dimensions with 1
    // Example:                          Reshape To:
    //   Input:   [1, 16, 32, 32]          Input:   [1, 16, 32, 32]
    //   Indices: [2, 5]                   Indices: [1, 10, 1, 1]
    //   Axis:    1                        Axis:    1
    //   Output:  [1, 2, 5, 32, 32]        Output:  [1, 10, 32, 32]

    auto indicesType = mlir::cast<NDTypeInterface>(indices.getType());
    auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    Shape newIndicesShape(inputType.getRank(), 1);
    newIndicesShape[axis] = indicesType.getShape().totalSize();
    auto reshapeIndicesOp = reshapeOperand(indices, newIndicesShape, takeOpLoc(origOp, "reshape_indices"));

    // HW requirement: each list entry must be 64 bits
    auto requiredType64 = mlir::IntegerType::get(origOp.getContext(), 64);

    // Convert non-4D shape to 4D for better hardware utilization
    // Only do this if the operation will benefit from multi-shave execution
    const auto reshapeIndicesType = mlir::cast<NDTypeInterface>(reshapeIndicesOp.getType());
    const auto reshapeIndicesShape = reshapeIndicesType.getShape();

    const bool shouldConvertTo4D =
            (reshapeIndicesShape.size() != 4) && VPU::shouldConvertUseMultiShaves(reshapeIndicesType);

    mlir::Value convertIndicesOp = [&]() -> mlir::Value {
        if (shouldConvertTo4D) {
            // Reshape to 4D: [1, totalSize, 1, 1] for non-4D inputs
            const auto totalSize = reshapeIndicesShape.totalSize();
            const Shape shape4D = {1, totalSize, 1, 1};
            const auto reshapeTo4DOp = reshapeOperand(reshapeIndicesOp, shape4D, takeOpLoc(origOp, "reshape_to_4d"));

            const auto convertOp = rewriter.createOrFold<VPU::ConvertOp>(origOp->getLoc(), reshapeTo4DOp,
                                                                         mlir::TypeAttr::get(requiredType64));

            // Reshape back to original shape
            return reshapeOperand(convertOp, reshapeIndicesShape, takeOpLoc(origOp, "reshape_from_4d"));
        } else {
            return rewriter.createOrFold<VPU::ConvertOp>(origOp->getLoc(), reshapeIndicesOp,
                                                         mlir::TypeAttr::get(requiredType64));
        }
    }();

    auto gatherDMAOp = rewriter.create<VPU::GatherDMAOp>(
            origOp.getLoc(), origOp.getInput(), convertIndicesOp, origOp.getAxis(), origOp.getAxisValueAttr(),
            origOp.getBatchDims(), /*multiClusterStrategy*/ nullptr, /*addressingMode*/ nullptr);

    // TODO (E#175972) Set ABSOLUTE Addressing mode when feature is enabled.
    // Until then will set default value as INDEXED addressing mode.
    // In order to set ABSOLUTE addressing mode we need to have tiling in order to satisfy HW requirements,
    // multiclustering and also tiling to fit in CMX done on the dim exactly after gather axis
    // (eg. for NCHW tensor : gather axis - C  tiling and multiclustering done on H)
    if (hasTilingDoneToBenefitAbsAddressing(origOp) && canHaveMulticlusteringToBenefitAbsAddressing(gatherDMAOp)) {
        gatherDMAOp.setAddressingMode(VPU::GatherAddressingMode::INDEXED);
    }
    auto reshapeOutOp =
            reshapeOperand(gatherDMAOp.getOutput(), outputType.getShape(), takeOpLoc(origOp, "reshape_output"));

    origOp.getOutput().replaceAllUsesWith(reshapeOutOp);
    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// MoveToDMAPass
//

class ConvertOpToDMAForPerformantExecutionPass final :
        public VPU::impl::ConvertOpToDMAForPerformantExecutionBase<ConvertOpToDMAForPerformantExecutionPass> {
public:
    explicit ConvertOpToDMAForPerformantExecutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertOpToDMAForPerformantExecutionPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::ConversionTarget adaptionTarget(ctx);

    adaptionTarget.addDynamicallyLegalOp<VPU::GatherOp>([&](VPU::GatherOp op) {
        if (VPU::isLegalConvertToGatherDMA(op, /*isElementTile*/ true, /*isIndicesTile*/ false, _log)) {
            return false;
        }
        if (VPU::isLegalConvertToGatherDMA(op, /*isElementTile*/ false, /*isIndicesTile*/ true, _log)) {
            return false;
        }
        return true;
    });
    adaptionTarget.addLegalOp<VPU::SliceOp>();
    adaptionTarget.addLegalOp<VPU::ConcatOp>();

    mlir::RewritePatternSet adaptionPatterns(&ctx);

    adaptionPatterns.add<TileGatherElement>(&ctx, _log);
    adaptionPatterns.add<TileGatherIndices>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, adaptionTarget, std::move(adaptionPatterns)))) {
        signalPassFailure();
        return;
    }

    const auto arch = config::getArch(func);
    // TODO: E#118296 Other ops and architectures will be enabled.
    if (arch >= config::ArchKind::NPU40XX) {
        mlir::ConversionTarget target(ctx);
        target.addDynamicallyLegalOp<VPU::GatherOp>([&](VPU::GatherOp op) {
            if (!VPU::isLegalConvertToGatherDMA(op, /*isElementTile*/ false, /*isIndicesTile*/ false, _log)) {
                return true;
            }
            return false;
        });

        target.addLegalOp<Const::DeclareOp>();
        target.addLegalOp<VPU::GatherDMAOp>();
        target.addLegalOp<VPU::ReshapeOp>();
        target.addLegalOp<VPU::ConvertOp>();

        mlir::RewritePatternSet patterns(&ctx);
        patterns.insert<MoveToDMAGather>(&ctx, _log);

        if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
}

}  // namespace

//
// createConvertOpToDMAForPerformantExecutionPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertOpToDMAForPerformantExecutionPass(Logger log) {
    return std::make_unique<ConvertOpToDMAForPerformantExecutionPass>(log);
}
