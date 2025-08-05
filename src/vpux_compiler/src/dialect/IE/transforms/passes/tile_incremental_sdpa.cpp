//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_TILEINCREMENTALSDPA
#define GEN_PASS_DEF_TILEINCREMENTALSDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class IncrementalSDPARewrite final : public mlir::OpRewritePattern<IE::IncrementalSDPAOp> {
public:
    IncrementalSDPARewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::IncrementalSDPAOp>(ctx), _log(log) {
        setDebugName("IncrementalSDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::IncrementalSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

int64_t align(int64_t value, int64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

mlir::Value createSlice(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value value, Dim dimension,
                        int64_t beginOffset, int64_t endOffset, const Logger& log) {
    auto shape = getShape(value);

    auto sliceOffset = Shape(shape.size(), 0);
    sliceOffset[dimension] = checked_cast<int64_t>(beginOffset);
    auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffset);

    auto sliceSize = Shape(shape);
    sliceSize[dimension] = endOffset - beginOffset;
    auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSize);

    log.trace("Created SliceOp with offset {0} and size {1} at {2}", sliceOffset, sliceSize, loc);
    return rewriter.create<IE::SliceOp>(loc, value, offsetsAttr, sizesAttr);
}

mlir::LogicalResult IncrementalSDPARewrite::matchAndRewrite(IE::IncrementalSDPAOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();

    const auto kvNumBlocksAttr = origOp.getKvNumBlocksAttr();
    VPUX_THROW_UNLESS(kvNumBlocksAttr != nullptr, "Expected to have kv_num_blocks attribute set to be able to tile {0}",
                      origOp->getName());

    const auto kvNumBlocks = parseIntAttr<int64_t>(kvNumBlocksAttr);

    // Clear the tiling attribute once to copy a clean operation
    const auto clearTilingAttr = [&] {
        origOp.setKvNumBlocksAttr(nullptr);
    };
    rewriter.modifyOpInPlace(origOp, clearTilingAttr);

    if (kvNumBlocks < 2) {
        log.trace("No need to tile the operation");
        return mlir::success();
    }

    // Tiling parameters
    auto keyShape = getShape(origOp.getKey());
    auto heightIndex = static_cast<int64_t>(keyShape.size() - 2);
    auto heightDim = Dim(heightIndex);
    auto sourceSeqLen = keyShape[heightDim];  // key and value share the same height value
    auto alignment = 1;                       // TODO: might affect performance

    // For attention_mask tiling
    auto widthIndex = static_cast<int64_t>(keyShape.size() - 1);
    auto widthDim = Dim(widthIndex);

    // Partial values that are chained through IncrementalSDPAOp
    auto max = origOp.getInputRunningMax();
    auto sum = origOp.getInputRunningSum();
    auto out = origOp.getInputPartialOutput();

    auto beginOffset = int64_t{0};
    auto i = int64_t{0};
    while (true) {
        log.trace("Unrolling {0} - {1} / {2} times", origOp->getName(), i + 1, kvNumBlocks);
        auto nextOffset = ((i + 1) * sourceSeqLen) / kvNumBlocks;
        auto alignedOffset = align(nextOffset, alignment);
        auto endOffset = std::min(alignedOffset, sourceSeqLen);

        auto keySlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "key_slice_{0}", i), origOp.getKey(),
                                    heightDim, beginOffset, endOffset, log);
        auto valueSlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "value_slice_{0}", i), origOp.getValue(),
                                      heightDim, beginOffset, endOffset, log);
        // TODO: slice an optional scale tensor.
        // Currently this tensor is created as a constant in the DecomposeIncrementalSDPA pass.

        mlir::IRMapping mapper;
        mapper.map(origOp.getKey(), keySlice);
        mapper.map(origOp.getValue(), valueSlice);

        if (origOp.getAttentionMask() != nullptr) {
            auto attentionMaskSlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "attention_mask_slice_{0}", i),
                                                  origOp.getAttentionMask(), widthDim, beginOffset, endOffset, log);

            mapper.map(origOp.getAttentionMask(), attentionMaskSlice);
        }

        mapper.map(origOp.getInputRunningMax(), max);
        mapper.map(origOp.getInputRunningSum(), sum);
        mapper.map(origOp.getInputPartialOutput(), out);

        auto* tiledOp = rewriter.clone(*origOp, mapper);
        auto tileLoc = appendLoc(origOp->getLoc(), "incremental_sdpa_{0}", i);
        tiledOp->setLoc(tileLoc);
        vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::ALL);
        log.trace("Unrolled {0} - {1}", tiledOp->getName(), tiledOp->getResult(0));

        if (i + 1 == kvNumBlocks) {
            rewriter.replaceOp(origOp, tiledOp);

            return mlir::success();
        }

        // Propagate intermediate values to the next IncrementalSDPAOp
        auto tiledIncrementalSdpa = mlir::cast<IE::IncrementalSDPAOp>(tiledOp);
        max = tiledIncrementalSdpa.getResultRunningMax();
        sum = tiledIncrementalSdpa.getResultRunningSum();
        out = tiledIncrementalSdpa.getResultPartialOutput();

        beginOffset = endOffset;
        i++;
    }
}

//
// TileIncrementalSDPA
//

class TileIncrementalSDPA final : public IE::impl::TileIncrementalSDPABase<TileIncrementalSDPA> {
public:
    explicit TileIncrementalSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void TileIncrementalSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    const auto isLegal = [](IE::IncrementalSDPAOp op) {
        return op.getKvNumBlocksAttr() == nullptr;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::IncrementalSDPAOp>(isLegal);
    target.addLegalOp<IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IncrementalSDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createTileIncrementalSDPAPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createTileIncrementalSDPAPass(Logger log) {
    return std::make_unique<TileIncrementalSDPA>(log);
}
