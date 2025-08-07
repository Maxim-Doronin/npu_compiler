//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/DebugInfo/LogicalView/Core/LVElement.h>
#include <mlir-c/AffineMap.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <cstdint>

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_UNROLLSPACETODEPTHDMA
#define GEN_PASS_DEF_UNROLLSPACETODEPTHDMA
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

//
// SpaceToDepthDMARewriter
//

class SpaceToDepthDMARewriter final : public mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp> {
public:
    SpaceToDepthDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(std::move(log)) {
        setDebugName("SpaceToDepthDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult matchAndRewriteClusterDMA(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                  mlir::PatternRewriter& rewriter) const;

private:
    void unroll(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask, mlir::PatternRewriter& rewriter) const;
    mlir::LogicalResult unrollSegmentedOrOverlapped(VPUIP::SpaceToDepthDMAOp spaceToDepthOp,
                                                    VPUIP::DistributedBufferType distributedType,
                                                    mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;

    // Minimum rank is 3, as defined by the opset
    static constexpr auto MIN_RANK = 3;
};

void SpaceToDepthDMARewriter::unroll(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                     mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    auto inDeclBuff = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto outDeclBuff = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    VPUX_THROW_WHEN(inType.getRank() < MIN_RANK, "Unrolling is supported only for rank {0} or higher shapes", MIN_RANK);
    VPUX_THROW_WHEN(outType.getRank() != inType.getRank(),
                    "Unrolling is not supported for input and output of different ranks");

    const auto blockSize = origOp.getBlockSize();
    const auto blocksFirst = origOp.getMode() == IE::SpaceToDepthMode::BLOCKS_FIRST;

    const auto buildTaskOp = [&](auto internalInputMemRef, auto inputBuffer, auto internalOutputMemRef,
                                 auto outputBuffer, auto internalInToOutMapping, int64_t dmaPort) {
        auto mappingOrder = mlir::AffineMapAttr::get(internalInToOutMapping);
        // After internal input and output representations have been obtained, the optimal loop order to obtain the
        // minimal number of DMA transactions can be computed/fetched from a cache.
        auto loopOrder = mlir::AffineMapAttr::get(
                mlir::AffineMap::getPermutationMap(VPUIP::getLinearMemOrder(internalInputMemRef), ctx));
        auto internalDataFlowAttr = VPUIP::InternalDataFlowAttr::get(ctx, internalInputMemRef, internalOutputMemRef,
                                                                     mappingOrder, loopOrder);

        VPURT::wrapIntoTaskOp<VPUIP::SpaceToDepthDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), inputBuffer,
                outputBuffer, vpux::getIntAttr(rewriter, dmaPort), origOp.getBlockSizeAttr(), origOp.getModeAttr(),
                nullptr, origOp.getIsOutOfOrderAttr(), origOp.getIsCriticalAttr(), origOp.getDmaHwpIdAttr(),
                origOp.getProfilingMetadataAttr(), internalDataFlowAttr);
    };

    // Before determining the internal representation of the data movement of a single transaction, we would need to
    // ensure the transaction can be executed by the DMA. Here we would need to query the DMA engine limits from the
    // DMAEngineLimits class. However, no actual unrolling for engine capabilities is done for now here as the maximum
    // transfer and stride level for NPU4+ is sufficient for all but the largest transfers (i.e. > 4 GB). For now, we
    // unroll solely to cover a target number of ports.

    VPUIP::splitSpaceToDepth(rewriter, buildTaskOp, vpurtTask, inType, inDeclBuff, outType, outDeclBuff, blockSize,
                             blocksFirst, _dmaPortCount);
}

mlir::LogicalResult SpaceToDepthDMARewriter::unrollSegmentedOrOverlapped(VPUIP::SpaceToDepthDMAOp spaceToDepthOp,
                                                                         VPUIP::DistributedBufferType distributedType,
                                                                         mlir::PatternRewriter& rewriter) const {
    auto loc = spaceToDepthOp->getLoc();
    auto ctx = spaceToDepthOp->getContext();

    const auto input = spaceToDepthOp.getInput();
    const auto output = spaceToDepthOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(spaceToDepthOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(distributedType.getCompactType());

    const auto distributionAttr = distributedType.getDistribution();
    const auto distMode = distributionAttr.getMode().getValue();

    VPUX_THROW_UNLESS(distMode == VPU::DistributionMode::SEGMENTED || distMode == VPU::DistributionMode::OVERLAPPED,
                      "Unsupported distribution mode: {0}", distMode);

    const auto blockSize = spaceToDepthOp.getBlockSize();

    const auto perClusterOutShapes = distributedType.getPerClusterMemoryShapes();
    const auto perClusterOutShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    const auto backInferInputShape = [&](ShapeRef outShape, int64_t blockSize) {
        auto inShape = Shape(outShape.raw());
        inShape[Dims4D::Act::H] *= blockSize;
        inShape[Dims4D::Act::W] *= blockSize;
        inShape[Dims4D::Act::C] /= (blockSize * blockSize);
        return inShape;
    };

    const auto origStrides = inputType.getStrides();
    const auto numClusters = perClusterOutShapes.size();

    SmallVector<NDTypeInterface> inTypes(numClusters);
    SmallVector<NDTypeInterface> outTypes(numClusters);
    for (auto clusterId : irange(numClusters)) {
        inTypes[clusterId] =
                inputType
                        .extractDenseTile(backInferInputShape(perClusterOutShapeOffsets[clusterId], blockSize),
                                          backInferInputShape(perClusterOutShapes[clusterId], blockSize))
                        .changeStrides(origStrides);
        outTypes[clusterId] =
                outputType.extractDenseTile(perClusterOutShapeOffsets[clusterId], perClusterOutShapes[clusterId]);
    }

    auto vpurtTask = spaceToDepthOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", spaceToDepthOp);

    rewriter.setInsertionPointAfter(vpurtTask);

    const auto getInputOperand = [&](mlir::Value operand, vpux::NDTypeInterface newType,
                                     mlir::Operation* insertionPoint, Byte offset) -> mlir::Value {
        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        Byte cmxOffset{declBuff.getByteOffset()};
        cmxOffset += offset;

        auto declBuffType = mlir::cast<vpux::NDTypeInterface>(declBuff.getType());
        VPUX_THROW_UNLESS(declBuffType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                          "Currently only support input in CMX");
        auto sectionIndex = declBuffType.getMemSpace().getIndex();
        VPUX_THROW_UNLESS(sectionIndex.has_value() && sectionIndex.value() == 0,
                          "Currently only support input in CMX0");

        auto section = declBuff.getSection();
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)), 0);
        newType = newType.changeMemSpace(symbolAttr);

        auto newDeclBuff = declBuff;

        auto memSpaceIndex = declBuffType.getMemSpace().getIndex();
        if (memSpaceIndex.has_value()) {
            newDeclBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                                  memSpaceIndex.value(), cmxOffset.count());
        } else {
            newDeclBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                                  cmxOffset.count());
        }

        return newDeclBuff;
    };

    const auto getOutputOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                      mlir::Operation* insertionPoint) -> mlir::Value {
        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
        auto newCMXType = newType.changeMemSpace(symbolAttr);

        return VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, insertionPoint, spaceToDepthOp->getLoc(), newCMXType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(ctx, ArrayRef({clusterId})), declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
    };

    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto tilingDim = Dim(VPU::getDistributedTilingAxis(tilingScheme));

    for (auto clusterId : irange(numClusters)) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto cmxOffset =
                perClusterOutShapeOffsets[clusterId][tilingDim] * static_cast<Byte>(newOutType.getStrides()[tilingDim]);
        const auto inputBuffer = getInputOperand(input, newInputType, inputInsertionPoint, cmxOffset);

        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOutputOperand(clusterId, output, newOutType, outputInsertionPoint);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        const auto dmaPort = clusterId % _dmaPortCount;
        auto newSpaceToDepthDMAOp = VPURT::wrapIntoTaskOp<VPUIP::SpaceToDepthDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
                vpux::getIntAttr(rewriter, dmaPort), spaceToDepthOp.getBlockSizeAttr(), spaceToDepthOp.getModeAttr(),
                /*dma_descriptor*/ nullptr, spaceToDepthOp.getIsOutOfOrderAttr(), spaceToDepthOp.getIsCriticalAttr(),
                spaceToDepthOp.getDmaHwpIdAttr(), spaceToDepthOp.getProfilingMetadataAttr(),
                /*InternalDataFlowAttr*/ nullptr);

        _log.trace("Insert new SpaceToDepthDMA: '{0}'", newSpaceToDepthDMAOp);
    }
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult SpaceToDepthDMARewriter::matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto outputType = spaceToDepthDMAOp.getOutputBuff().getType();
    if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType)) {
        return matchAndRewriteClusterDMA(spaceToDepthDMAOp, rewriter);
    }

    _log.trace("Get SpaceToDepthDMAOp : {0}", spaceToDepthDMAOp->getLoc());

    if (spaceToDepthDMAOp.getInternalDataFlow().has_value()) {
        _log.trace("This SpaceToDepthDMAOp has already been unrolled.");
        return mlir::failure();
    }

    auto vpurtTask = spaceToDepthDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.setInsertionPointAfter(vpurtTask);

    _log.trace("Unroll SpaceToDepthDMAOp {0}", spaceToDepthDMAOp->getLoc());

    unroll(spaceToDepthDMAOp, vpurtTask, rewriter);

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult SpaceToDepthDMARewriter::matchAndRewriteClusterDMA(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got SpaceToDepthDMA with DistributedType: {0}", spaceToDepthDMAOp);

    const auto input = spaceToDepthDMAOp.getInput();
    const auto output = spaceToDepthDMAOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    VPUX_THROW_UNLESS(inputType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
                              outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                      "Unexpected memory space: input {0}, output {1}", inputType.getMemoryKind(),
                      outputType.getMemoryKind());

    const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);
    VPUX_THROW_WHEN(distributedType == nullptr, "Expect distributed type for SpaceToDepthDMA op output, but got: {0}",
                    outputType);

    const auto distributionAttr = distributedType.getDistribution();
    VPUX_THROW_WHEN(distributionAttr == nullptr, "Failed to extract distributon attribute from distributed type.");

    const auto modeAttr = distributionAttr.getMode();
    VPUX_THROW_WHEN(modeAttr == nullptr, "Failed to extract mode from distribution attribute.");
    const auto mode = modeAttr.getValue();

    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                      "Unsupported distribution mode: {0}", modeAttr);
    return unrollSegmentedOrOverlapped(spaceToDepthDMAOp, distributedType, rewriter);
}

//
// UnrollSpaceToDepthDMAPass
//

class UnrollSpaceToDepthDMAPass final :
        public VPUIP::arch40xx::impl::UnrollSpaceToDepthDMABase<UnrollSpaceToDepthDMAPass> {
public:
    explicit UnrollSpaceToDepthDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollSpaceToDepthDMAPass::safeRunOnFunc() {
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollSpaceToDepthDMAPass)) {
        return;
    }
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SpaceToDepthDMARewriter>(&ctx, dmaPortCount, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollSpaceToDepthDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createUnrollSpaceToDepthDMAPass(Logger log) {
    return std::make_unique<UnrollSpaceToDepthDMAPass>(log);
}
