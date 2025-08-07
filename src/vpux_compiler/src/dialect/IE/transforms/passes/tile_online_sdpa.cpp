//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_TILEONLINESDPA
#define GEN_PASS_DEF_TILEONLINESDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class OnlineSDPARewrite final : public mlir::OpRewritePattern<IE::OnlineSDPAOp> {
public:
    OnlineSDPARewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::OnlineSDPAOp>(ctx), _log(log) {
        setDebugName("OnlineSDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::OnlineSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

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

mlir::LogicalResult OnlineSDPARewrite::matchAndRewrite(IE::OnlineSDPAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();

    // TODO: Figure out a way to compute Q and KV tiling.
    // There are multiple ways to select the tiling sizes.
    // The goal is to choose an optimal way to tile the tensors to reach the best performance.
    // All operations in the resulting subgraph should fit into CMX.
    // Operations should have [1, 1, 1, 1] isolated tiling strategy, because they should be already small enough.
    // Vertical fusion should not increase the strategy when doing the fusion.
    // Usual way of choosing Q and KV is to have Q == KV, using some kind of a formula dependent on CMX size.

    // TODO: Tiling parameters are just hardcoded, need to use a heuristic
    auto qNumBlocks = int64_t{2};
    auto kvNumBlocks = int64_t{2};
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (const auto qNumBlocksEnv = std::getenv("NPU_Q_NUM_BLOCKS")) {
        qNumBlocks = std::atoi(qNumBlocksEnv);
    }
    if (const auto kvNumBlocksEnv = std::getenv("NPU_KV_NUM_BLOCKS")) {
        kvNumBlocks = std::atoi(kvNumBlocksEnv);
    }
#endif
    log.trace("Computed tiling: Query - {0} times, KeyValue - {1} times", qNumBlocks, kvNumBlocks);

    // Save KeyValue tiling attribute to propagate the tiling decision to each IncrementalSdpaOp
    const auto setKvNumBlocks = [&] {
        origOp.setKvNumBlocks(kvNumBlocks);
    };
    rewriter.modifyOpInPlace(origOp, setKvNumBlocks);

    if (qNumBlocks < 2) {
        log.trace("No need to tile the operation");
        return mlir::success();
    }

    auto queryShape = getShape(origOp.getQuery());
    auto heightIndex = static_cast<int64_t>(queryShape.size() - 2);
    auto heightDim = Dim(heightIndex);
    auto height = queryShape[heightDim];
    auto alignment = 1;  // TODO: might affect performance

    SmallVector<mlir::Value> slices;

    auto beginOffset = int64_t{0};
    for (int64_t i : irange(qNumBlocks)) {
        log.trace("Tiling {0} - {1} / {2} times", origOp->getName(), i + 1, qNumBlocks);

        auto nextOffset = ((i + 1) * height) / qNumBlocks;
        auto alignedOffset = align(nextOffset, alignment);
        auto endOffset = std::min(alignedOffset, height);

        auto querySlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "query_slice_{0}", i), origOp.getQuery(),
                                      heightDim, beginOffset, endOffset, log);

        mlir::IRMapping mapper;
        mapper.map(origOp.getQuery(), querySlice);

        if (origOp.getAttentionMask() != nullptr) {
            auto attentionMaskSlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "attention_mask_slice_{0}", i),
                                                  origOp.getAttentionMask(), heightDim, beginOffset, endOffset, log);

            mapper.map(origOp.getAttentionMask(), attentionMaskSlice);
        }

        auto* tiledOp = rewriter.clone(*origOp, mapper);
        auto tileLoc = appendLoc(origOp->getLoc(), "output_slice_{0}", i);
        tiledOp->setLoc(tileLoc);
        vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::ALL);
        log.trace("Tiled {0} - {1}", tiledOp->getName(), tiledOp->getResult(0));

        slices.push_back(tiledOp->getResult(0));

        beginOffset = endOffset;
    }

    auto concat = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "concat_online_sdpa"), slices, heightDim);

    rewriter.replaceOp(origOp, concat);

    return mlir::success();
}

//
// TileOnlineSDPA
//

class TileOnlineSDPA final : public IE::impl::TileOnlineSDPABase<TileOnlineSDPA> {
public:
    explicit TileOnlineSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void TileOnlineSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    // Compute Q and KV tiling strategy and use it as a legality reason
    const auto isLegal = [](IE::OnlineSDPAOp op) {
        return op.getKvNumBlocksAttr() != nullptr;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::OnlineSDPAOp>(isLegal);
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OnlineSDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createTileOnlineSDPAPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createTileOnlineSDPAPass(Logger log) {
    return std::make_unique<TileOnlineSDPA>(log);
}
