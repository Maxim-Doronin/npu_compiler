//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <map>
#include <optional>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_FLASHSDPATILING
#define GEN_PASS_DEF_FLASHSDPATILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// Explicit tiling strategy with semantic dimension names
struct FlashSDPATilingStrategy {
    int64_t headTiles{1};      // Number of tiles on the Heads (C) dimension
    int64_t querySeqTiles{1};  // Number of tiles on Query sequence length (H)
    int64_t kvNumBlocks{1};    // Number of KV sequence unrolls (propagated to UnrollFlashSDPA)
};

class FlashSDPATilingRewrite final : public mlir::OpRewritePattern<VPU::FlashSDPAOp> {
public:
    FlashSDPATilingRewrite(mlir::MLIRContext* ctx, bool enablePipelining, Logger log)
            : mlir::OpRewritePattern<VPU::FlashSDPAOp>(ctx), _enablePipelining(enablePipelining), _log(log) {
        setDebugName("FlashSDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FlashSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enablePipelining = true;
    Logger _log;
};

// Estimate if we apply isolated tiling with the given Query number of slices
// how many times we would have to tile on Key/Value to fit everything in CMX
std::optional<int64_t> estimateRequiredKvNumBlocks(VPU::FlashSDPAOp origOp, ArrayRef<NDTypeInterface> tiledTensors,
                                                   Byte reservedMem) {
    if (origOp.fitIntoCMXAfterKeyValueTiling(tiledTensors, reservedMem, /*kvNumBlocks*/ 1)) {
        return 1;
    }

    const auto keyShape = getShape(origOp.getKey());
    const auto sourceSeqLen = keyShape[Dims4D::Act::H];

    const auto keyType = mlir::cast<NDTypeInterface>(origOp.getKey().getType());
    const auto elemType = keyType.getElementType();
    const auto alignment = vpux::VPU::NCEInvariant::getAlignment(elemType);

    auto kvNumBlocks = int64_t{1};
    auto dimSize = sourceSeqLen;
    while (dimSize > alignment) {
        kvNumBlocks = divUp(sourceSeqLen, dimSize - alignment);
        dimSize = alignValUp(divUp(sourceSeqLen, kvNumBlocks), alignment);

        if (origOp.fitIntoCMXAfterKeyValueTiling(tiledTensors, reservedMem, kvNumBlocks)) {
            return kvNumBlocks;
        }
    }

    // No split on Key/Value was found that would fit CMX
    return std::nullopt;
}

// Operand indices that are pipelined (duplicated) across tiled operations.
// These need reserved CMX memory for double-buffering during tiling estimation.
SmallVector<int64_t> getPipelinedBufferIndices(bool hasAttentionMask) {
    auto indices = SmallVector<int64_t>{
            0,   // query
            7,   // input_running_output
            8,   // input_running_max
            9,   // input_running_sum
            10,  // attention_mask (optional)
            11,  // result_running_output
            12,  // result_running_max
            13,  // result_running_sum
    };

    if (!hasAttentionMask) {
        indices.pop_back();
    }

    return indices;
}

// Estimate the query sequence tiling and KV blocks needed for a given head tile size,
// always reserving memory for query pipelining.
//
// CMX layout for 1 operation with pipelining:
// [ ======== Total CMX Size ======== ]        v-- do not duplicate Shared buffers for pipelined ops
// [ Shared | Pipelined0 | Pipelined1 ] + [ Shared ]
// [     CMX for 1 Op    ]     ^-- reserved CMX
std::optional<FlashSDPATilingStrategy> estimateTiling(VPU::FlashSDPAOp origOp, bool enablePipelining, Logger log) {
    // Unroll to 1 head per op
    const auto resultShape = getShape(origOp.getResultRunningOutput());
    const auto qHeads = resultShape[Dims4D::Act::C];
    const auto headTiles = qHeads;

    const auto alignment = getAlignment(origOp, {}, {});

    auto tiledResultShape = Shape(resultShape);
    const auto headDimSize = resultShape[Dims4D::Act::C];
    tiledResultShape[Dims4D::Act::C] = alignValUp(divUp(headDimSize, headTiles), alignment[Dims4D::Act::C.ind()]);

    // Now try to find query sequence tiling that fits CMX
    const auto seqLenDimSize = resultShape[Dims4D::Act::H];
    const auto seqAlignment = alignment[Dims4D::Act::H.ind()];

    auto querySeqTiles = int64_t{1};
    auto curSeqSize = seqLenDimSize;
    auto kvNumBlocks = std::optional<int64_t>{};
    auto swOp = mlir::cast<VPU::SWOpInterface>(origOp.getOperation());

    while (true) {
        auto tiledTensorTypes = getAllOperandsSwInterface(swOp, TileInfo{tiledResultShape}, log);

        const auto hasAttentionMask = (static_cast<int64_t>(tiledTensorTypes.size()) == 14);
        auto pipelinedBuffersIndices = getPipelinedBufferIndices(hasAttentionMask);

        auto reservedMemory = Byte{0};
        if (enablePipelining) {
            for (auto index : pipelinedBuffersIndices) {
                reservedMemory += tiledTensorTypes[index].getTotalAllocSize();
            }
        }

        kvNumBlocks = estimateRequiredKvNumBlocks(origOp, tiledTensorTypes, reservedMemory);

        // Optimal strategy is to not tile on Key/Value tensors if possible
        if (kvNumBlocks.has_value() && kvNumBlocks.value() == 1) {
            break;
        }

        if (curSeqSize <= seqAlignment) {
            break;
        }

        querySeqTiles = divUp(seqLenDimSize, curSeqSize - seqAlignment);
        curSeqSize = alignValUp(divUp(seqLenDimSize, querySeqTiles), seqAlignment);

        tiledResultShape[Dims4D::Act::H] = curSeqSize;
    }

    if (!kvNumBlocks.has_value()) {
        return std::nullopt;
    }

    return FlashSDPATilingStrategy{headTiles, querySeqTiles, kvNumBlocks.value()};
}

mlir::LogicalResult applyTileStrategyFlashSDPA(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                               mlir::RewriterBase& rewriter, Logger log) {
    const auto results = origOp->getResults();

    auto resultTileValues = SmallVector<SmallVector<mlir::Value>>(results.size());
    auto resultTileOffsets = SmallVector<SmallVector<Shape>>(results.size());

    // Cache tiled inputs by (operandIndex, tile shape, tile offsets).
    // If two output tiles produce the same input slice for an operand, reuse it.
    using TileCacheKey = SmallVector<int64_t>;
    auto tiledInputCache = std::map<TileCacheKey, mlir::Value>();

    for (const auto& outputTile : tiles) {
        auto inputTiling = origOp.backInferTileInfo(outputTile, log);
        auto& inTiles = inputTiling.tiles;

        VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

        mlir::IRMapping mapper;
        for (auto [inputIdx, origInput] : origOp->getOperands() | indexed) {
            const auto& inTile = inTiles[inputIdx];

            auto cacheKey = SmallVector<int64_t>{static_cast<int64_t>(inputIdx)};
            cacheKey.append(inTile.shape.begin(), inTile.shape.end());
            cacheKey.append(inTile.offsets.begin(), inTile.offsets.end());

            auto it = tiledInputCache.find(cacheKey);
            mlir::Value tiledInput;
            if (it != tiledInputCache.end()) {
                tiledInput = it->second;
            } else {
                const auto valName = printToString("input {0}", inputIdx);
                tiledInput = vpux::VPU::makeTile(rewriter, origOp->getLoc(), origInput, inTile, valName);
                tiledInputCache[cacheKey] = tiledInput;
            }

            mapper.map(origInput, tiledInput);
        }

        const auto tileLoc = appendLoc(origOp->getLoc(), "output tile {0}", outputTile.offsets);

        auto* tiledOp = rewriter.clone(*origOp, mapper);
        tiledOp->setLoc(tileLoc);

        auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(tiledOp);
        VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                        tiledOp->getName());

        tiledBuilderOp.adjustAttrs(inputTiling, outputTile);

        vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::ALL);

        auto tiledResults = tiledOp->getResults();

        const auto outputTiling = origOp.getOutputTiling(outputTile, log);
        VPUX_THROW_UNLESS(results.size() == outputTiling.size(),
                          "Number of results '{0}' doesn't match with number of output tiles '{1}' at '{2}'",
                          results.size(), outputTiling.size(), origOp->getLoc());

        for (const auto i : irange(results.size())) {
            const auto& outputTile = outputTiling[i];
            auto tiledResult = tiledResults[i];

            const auto tiledShape = getShape(tiledResult);
            VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                              "Inferred output shape '{0}' doesn't match tiled shape '{1}' at '{2}'", tiledShape,
                              outputTile.shape, tiledResult.getDefiningOp()->getLoc());

            const auto resultType = mlir::cast<vpux::NDTypeInterface>(results[i].getType());
            const auto resultDenseTile = resultType.extractDenseTile(outputTile.offsets, outputTile.shape);

            tiledResult.setType(resultDenseTile);

            copyLoopAttributes(origOp, tiledResult.getDefiningOp());

            resultTileValues[i].push_back(tiledResult);
            resultTileOffsets[i].push_back(outputTiling[i].offsets);
        }
    }

    SmallVector<mlir::Value> concatOps;
    for (const auto i : irange(results.size())) {
        auto resultType = origOp->getResult(i).getType();
        auto tileValues = mlir::ValueRange(resultTileValues[i]);
        auto tileOffsets = ArrayRef(resultTileOffsets[i]);

        auto concatOp = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), resultType, tileValues, tileOffsets);

        concatOps.push_back(concatOp.getOutput());
    }

    rewriter.replaceOp(origOp, concatOps);

    return mlir::success();
}

mlir::LogicalResult FlashSDPATilingRewrite::matchAndRewrite(VPU::FlashSDPAOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();

    auto resultShape = getShape(origOp.getResult(0));
    auto resultRank = resultShape.size();
    if (resultRank < 2) {
        return errorAt(origOp, "Output shape must at least have a rank 2, got {0}", resultRank);
    }

    auto strategy = estimateTiling(origOp, _enablePipelining, log);
    if (!strategy.has_value()) {
        return errorAt(origOp, "Failed to estimate tiling for FlashSDPA operation. Tensors will not fit CMX.");
    }

    log.trace("Computed tiling: heads={0}, querySeq={1}, kv={2}", strategy->headTiles, strategy->querySeqTiles,
              strategy->kvNumBlocks);

    // Save Key/Value tiling attribute to propagate the tiling decision to each FlashSDPAOp.
    rewriter.modifyOpInPlace(origOp, [&] {
        origOp.setKvNumBlocks(strategy->kvNumBlocks);
    });

    // Build the tiling divisor shape: [N=1, C=headTiles, H=querySeqTiles, W=1]
    const auto firstOutputShape = getShape(origOp.getResultRunningOutput());
    auto tilingStrategy = Shape(firstOutputShape.size(), 1);
    tilingStrategy[Dims4D::Act::N] = firstOutputShape[Dims4D::Act::N];  // Unroll on Batch
    tilingStrategy[Dims4D::Act::C] = strategy->headTiles;
    tilingStrategy[Dims4D::Act::H] = strategy->querySeqTiles;

    const auto alignment = getAlignment(origOp, {}, {});
    const auto unrollSpatialFirst = true;
    const auto firstOutputTiles = fillDividedTiles(tilingStrategy, firstOutputShape, alignment, unrollSpatialFirst);

    if (mlir::failed(firstOutputTiles)) {
        return errorAt(origOp,
                       "Failed to compute tiling for output shape: '{0}', tiling strategy: '{1}', alignment: '{2}'",
                       firstOutputShape, tilingStrategy, alignment);
    }

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    origOp->getName());

    auto result = applyTileStrategyFlashSDPA(tilingBuilder, firstOutputTiles.value(), rewriter, log);
    if (mlir::failed(result)) {
        return errorAt(origOp, "Failed to rewrite original operation with the tiled one");
    }

    return mlir::success();
}

//
// FlashSDPATiling
//

class FlashSDPATiling final : public VPU::impl::FlashSDPATilingBase<FlashSDPATiling> {
public:
    explicit FlashSDPATiling(bool enablePipelining, Logger log): _enablePipelining(enablePipelining) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    bool _enablePipelining = true;
};

mlir::LogicalResult FlashSDPATiling::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enablePipelining.hasValue()) {
        _log.trace("Overloading FlashSDPATiling enablePipelining argument by MLIR variable");
        _enablePipelining = enablePipelining.getValue();
    }
    return mlir::success();
}

void FlashSDPATiling::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    // Mark each operation with a tiling loop index for loop-allocation scheduling
    func->walk([tilingIndex = 0ll](VPU::FlashSDPAOp flashSdpa) mutable {
        flashSdpa->setAttr(TILING_LOOP_INDEX_ATTR_NAME, TilingLoopIndexAttr::get(flashSdpa->getContext(), tilingIndex));
        ++tilingIndex;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FlashSDPATilingRewrite>(&ctx, _enablePipelining, _log);

    mlir::walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFlashSDPATilingPass
//

std::unique_ptr<mlir::Pass> VPU::createFlashSDPATilingPass(bool enablePipelining, Logger log) {
    return std::make_unique<FlashSDPATiling>(enablePipelining, log);
}
