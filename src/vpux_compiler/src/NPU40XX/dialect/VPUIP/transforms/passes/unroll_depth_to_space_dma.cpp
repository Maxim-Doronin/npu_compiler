//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_UNROLLDEPTHTOSPACEDMA
#define GEN_PASS_DEF_UNROLLDEPTHTOSPACEDMA
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

//
// DepthToSpaceDMARewriter
//

class DepthToSpaceDMARewriter final : public mlir::OpRewritePattern<VPUIP::DepthToSpaceDMAOp> {
public:
    DepthToSpaceDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::DepthToSpaceDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(std::move(log)) {
        setDebugName("DepthToSpaceDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult matchAndRewriteClusterDMA(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                  mlir::PatternRewriter& rewriter) const;

private:
    auto adjustChannelsForPadding(vpux::NDTypeInterface ndType, int64_t paddedChannels) const;
    void unroll(VPUIP::DepthToSpaceDMAOp origOp, vpux::VPURT::TaskOp vpurtTask, mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;

    // Minimum rank is 3, as defined by the opset
    static constexpr auto MIN_RANK = 3;
    // Channel dim index is always 1, as defined by the opset
    static constexpr auto C_DIM_INDEX = 1;
};

auto DepthToSpaceDMARewriter::adjustChannelsForPadding(vpux::NDTypeInterface ndType, int64_t paddedChannels) const {
    auto shape = ndType.getShape().toValues();
    auto strides = ndType.getStrides();

    VPUX_THROW_WHEN(shape[Dim(C_DIM_INDEX)] <= paddedChannels, "Too many padded channels");

    shape[Dim(C_DIM_INDEX)] -= paddedChannels;

    // Use new shape
    ndType = ndType.changeShape(shape);
    // Keep old strides
    ndType = ndType.changeStrides(strides);

    return ndType;
}

void DepthToSpaceDMARewriter::unroll(VPUIP::DepthToSpaceDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                     mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    VPUX_THROW_WHEN(inType.getRank() < MIN_RANK, "Unrolling is supported only for rank {0} or higher shapes", MIN_RANK);
    VPUX_THROW_WHEN(outType.getRank() != inType.getRank(),
                    "Unrolling is not supported for input and output of different ranks");
    VPUX_THROW_WHEN(inType.getCompactAllocSize() != outType.getCompactAllocSize(),
                    "Size mismatch between input and output");

    auto inDeclBuff = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto outDeclBuff = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto paddedICAttr = origOp.getPaddedChannels() ? origOp.getPaddedChannels().value().getInput() : nullptr;
    auto paddedOCAttr = origOp.getPaddedChannels() ? origOp.getPaddedChannels().value().getOutput() : nullptr;

    if (paddedICAttr != nullptr && paddedOCAttr != nullptr) {
        inType = adjustChannelsForPadding(inType, paddedICAttr.getInt());
        outType = adjustChannelsForPadding(outType, paddedOCAttr.getInt());
    }

    const auto blockSize = origOp.getBlockSize();
    const auto blocksFirst = origOp.getMode() == IE::DepthToSpaceMode::BLOCKS_FIRST;

    const auto buildTaskOp = [&](auto internalOutputMemRef, auto outputBuffer, auto internalInputMemRef,
                                 auto inputBuffer, auto internalInToOutMapping, int64_t dmaPort) {
        // Invert SpaceToDepth mapping order
        auto mappingOrder = mlir::AffineMapAttr::get(mlir::inversePermutation(internalInToOutMapping));
        // After internal input and output representations have been obtained, the optimal loop order to obtain the
        // minimal number of DMA transactions can be computed/fetched from a cache.
        auto loopOrder = mlir::AffineMapAttr::get(
                mlir::AffineMap::getPermutationMap(VPUIP::getLinearMemOrder(internalInputMemRef), ctx));
        auto internalDataFlowAttr = VPUIP::InternalDataFlowAttr::get(ctx, internalInputMemRef, internalOutputMemRef,
                                                                     mappingOrder, loopOrder);

        VPURT::wrapIntoTaskOp<VPUIP::DepthToSpaceDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), inputBuffer,
                outputBuffer, vpux::getIntAttr(rewriter, dmaPort), origOp.getBlockSizeAttr(), origOp.getModeAttr(),
                nullptr, nullptr, origOp.getIsOutOfOrderAttr(), origOp.getIsCriticalAttr(), origOp.getDmaHwpIdAttr(),
                origOp.getProfilingMetadataAttr(), internalDataFlowAttr);
    };

    // Before determining the internal representation of the data movement of a single transaction, we would need to
    // ensure the transaction can be executed by the DMA. Here we would need to query the DMA engine limits from the
    // DMAEngineLimits class. However, no actual unrolling for engine capabilities is done for now here as the maximum
    // transfer and stride level for NPU4+ is sufficient for all but the largest transfers (i.e. > 4 GB). For now, we
    // unroll solely to cover a target number of ports.

    // Use SpaceToDepth helper
    VPUIP::splitSpaceToDepth(rewriter, buildTaskOp, vpurtTask, outType, outDeclBuff, inType, inDeclBuff, blockSize,
                             blocksFirst, _dmaPortCount);
}

mlir::LogicalResult DepthToSpaceDMARewriter::matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto inputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(depthToSpaceDMAOp.getInput().getType());
    auto outputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(depthToSpaceDMAOp.getOutputBuff().getType());

    if (inputType != nullptr || outputType != nullptr) {
        return matchAndRewriteClusterDMA(depthToSpaceDMAOp, rewriter);
    }
    _log.trace("Got DepthToSpaceDMAOp: {0}", depthToSpaceDMAOp->getLoc());

    auto vpurtTask = depthToSpaceDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    if (depthToSpaceDMAOp.getInternalDataFlow().has_value()) {
        _log.nest().trace("This DepthToSpaceDMAOp has already been unrolled.");
        return mlir::failure();
    }

    unroll(depthToSpaceDMAOp, vpurtTask, rewriter);

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult DepthToSpaceDMARewriter::matchAndRewriteClusterDMA(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got DepthToSpaceDMA with DistributedBuffer: {0}", depthToSpaceDMAOp->getLoc());

    auto ctx = depthToSpaceDMAOp->getContext();

    const auto blockSize = depthToSpaceDMAOp.getBlockSize();
    int64_t paddedIC = 0;
    int64_t paddedOC = 0;

    if (depthToSpaceDMAOp.getPaddedChannels().has_value()) {
        paddedIC = depthToSpaceDMAOp.getPaddedChannels().value().getInput()
                           ? depthToSpaceDMAOp.getPaddedChannels().value().getInput().getInt()
                           : 0;
        paddedOC = depthToSpaceDMAOp.getPaddedChannels().value().getOutput()
                           ? depthToSpaceDMAOp.getPaddedChannels().value().getOutput().getInt()
                           : 0;
    }

    const auto input = depthToSpaceDMAOp.getInput();
    const auto output = depthToSpaceDMAOp.getOutputBuff();

    const auto distributedInputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(input.getType());
    const auto distributedOutputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());

    VPUX_THROW_WHEN(distributedInputType == nullptr && distributedOutputType == nullptr,
                    "At least one of operands must have DistributedBuffer type");

    const auto getDistModeAttr = [&](VPUIP::DistributedBufferType distType) {
        const auto distAttr = distType.getDistribution();
        VPUX_THROW_WHEN(distAttr == nullptr, "Failed to extract distribution tensor from distributed type");
        return distAttr.getMode();
    };

    if (distributedInputType != nullptr) {
        const auto inputDistModeAttr = getDistModeAttr(distributedInputType);
        VPUX_THROW_UNLESS(
                inputDistModeAttr != nullptr && inputDistModeAttr.getValue() == VPU::DistributionMode::SEGMENTED,
                "Unsupported input distributed mode: {0}", inputDistModeAttr);
    }

    if (distributedOutputType != nullptr) {
        const auto outputDistModeAttr = getDistModeAttr(distributedOutputType);
        VPUX_THROW_UNLESS(
                outputDistModeAttr != nullptr && outputDistModeAttr.getValue() == VPU::DistributionMode::SEGMENTED,
                "Unsupported output distributed mode: {0}", outputDistModeAttr);
    }

    auto vpurtTask = depthToSpaceDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", depthToSpaceDMAOp);

    const auto inferOutputShape = [&](ShapeRef inShape) {
        auto outShape = Shape(inShape.raw());
        outShape[Dims4D::Act::H] *= blockSize;
        outShape[Dims4D::Act::W] *= blockSize;
        outShape[Dims4D::Act::C] = (outShape[Dims4D::Act::C] - paddedIC) / (blockSize * blockSize) + paddedOC;
        return outShape;
    };

    const auto loc = depthToSpaceDMAOp->getLoc();

    mlir::SmallVector<mlir::Value> inputBuffers;
    mlir::SmallVector<mlir::Value> outputBuffers;

    if (distributedInputType != nullptr && distributedOutputType != nullptr) {
        _log.nest().trace("Got multi-cluster to multi-cluster case");
        const auto inputPerClusterShapes = distributedInputType.getPerClusterMemoryShapes();
        const auto outputPerClusterShapes = distributedOutputType.getPerClusterMemoryShapes();

        const auto isShapeCompatible = [&](ShapeRef inShape, ShapeRef outShape) {
            return inShape == VPUIP::backInferD2SInputShape(outShape.toValues(), paddedOC, paddedIC, blockSize);
        };

        VPUX_THROW_UNLESS(llvm::all_of_zip(inputPerClusterShapes, outputPerClusterShapes, isShapeCompatible),
                          "Shape per cluster not compatible");

        const auto numClusters = checked_cast<int64_t>(inputPerClusterShapes.size());

        inputBuffers = VPUIP::getPerClusterMemoryBuffers(ctx, loc, "input", input, numClusters, rewriter);
        outputBuffers = VPUIP::getPerClusterMemoryBuffers(ctx, loc, "output", output, numClusters, rewriter);
    }

    if (distributedInputType != nullptr && distributedOutputType == nullptr) {
        _log.nest().trace("Got multi-cluster to single-cluster case");
        const auto outputShapes = SmallVector<vpux::Shape>(
                llvm::map_range(distributedInputType.getPerClusterMemoryShapes(), inferOutputShape));
        const auto outputShapeOffsets = SmallVector<vpux::Shape>(
                llvm::map_range(distributedInputType.getPerClusterMemoryShapeOffsets(), inferOutputShape));

        const auto numClusters = checked_cast<int64_t>(outputShapes.size());

        inputBuffers = VPUIP::getPerClusterMemoryBuffers(ctx, loc, "input", input, numClusters, rewriter);
        outputBuffers = VPUIP::getSplitBuffers(ctx, loc, "output", output, outputShapes, outputShapeOffsets,
                                               numClusters, rewriter);
    }

    if (distributedInputType == nullptr && distributedOutputType != nullptr) {
        _log.nest().trace("Got single-cluster to multi-cluster case");

        const auto inputShapes = SmallVector<Shape>(
                llvm::map_range(distributedOutputType.getPerClusterMemoryShapes(), [&](ShapeRef outShape) {
                    return VPUIP::backInferD2SInputShape(outShape.toValues(), paddedOC, paddedIC, blockSize);
                }));

        const auto inputShapeOffsets = SmallVector<Shape>(
                llvm::map_range(distributedOutputType.getPerClusterMemoryShapeOffsets(), [&](ShapeRef outShape) {
                    return VPUIP::backInferD2SInputShape(outShape.toValues(), paddedOC, paddedIC, blockSize);
                }));

        const auto numClusters = checked_cast<int64_t>(inputShapes.size());
        inputBuffers =
                VPUIP::getSplitBuffers(ctx, loc, "input", input, inputShapes, inputShapeOffsets, numClusters, rewriter);
        outputBuffers = VPUIP::getPerClusterMemoryBuffers(ctx, loc, "output", output, numClusters, rewriter);
    }

    VPUX_THROW_WHEN(inputBuffers.size() != outputBuffers.size(), "Size of input/output buffers list must match");
    const auto numClusters = inputBuffers.size();

    rewriter.setInsertionPointAfter(vpurtTask);

    int64_t dmaPort = 0;
    for (size_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(depthToSpaceDMAOp->getLoc(), "_cluster_{0}", clusterId);
        auto newDepthToSpaceDMAOp = VPURT::wrapIntoTaskOp<VPUIP::DepthToSpaceDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffers[clusterId],
                outputBuffers[clusterId], vpux::getIntAttr(rewriter, dmaPort), depthToSpaceDMAOp.getBlockSizeAttr(),
                depthToSpaceDMAOp.getModeAttr(), nullptr, depthToSpaceDMAOp.getPaddedChannelsAttr(),
                depthToSpaceDMAOp.getIsOutOfOrderAttr(), depthToSpaceDMAOp.getIsCriticalAttr(),
                depthToSpaceDMAOp.getDmaHwpIdAttr(), depthToSpaceDMAOp.getProfilingMetadataAttr(),
                /*internalDataFlow= */ nullptr);

        dmaPort = (dmaPort + 1) % _dmaPortCount;

        _log.nest().trace("Insert new DepthToSpaceDMAOp: '{0}'", newDepthToSpaceDMAOp);
    }
    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

//
// UnrollDepthToSpaceDMAPass
//

class UnrollDepthToSpaceDMAPass final :
        public VPUIP::arch40xx::impl::UnrollDepthToSpaceDMABase<UnrollDepthToSpaceDMAPass> {
public:
    explicit UnrollDepthToSpaceDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollDepthToSpaceDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollDepthToSpaceDMAPass)) {
        return;
    }
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DepthToSpaceDMARewriter>(&ctx, dmaPortCount, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollDepthToSpaceDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createUnrollDepthToSpaceDMAPass(Logger log) {
    return std::make_unique<UnrollDepthToSpaceDMAPass>(log);
}
