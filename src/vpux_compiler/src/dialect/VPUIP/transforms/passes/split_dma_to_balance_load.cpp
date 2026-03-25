//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <optional>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SPLITDMATOBALANCELOAD
#define GEN_PASS_DEF_SPLITDMATOBALANCELOAD
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

using BuffersPair = std::pair<mlir::Value, mlir::Value>;

Shape getSplitShape(NDTypeInterface bufferType, vpux::Dim tileDim, int64_t newDimSize) {
    Shape subShape = bufferType.getShape().toValues();
    subShape[tileDim] = newDimSize;
    return subShape;
}

NDTypeInterface getNewBufferType(NDTypeInterface bufferType, vpux::Dim tileDim, int64_t dimOffset, int64_t newDimSize) {
    const auto newShape = getSplitShape(bufferType, tileDim, newDimSize);
    Shape newOffset(SmallVector<int64_t>(newShape.size(), 0));
    newOffset[tileDim] = dimOffset;

    NDTypeInterface newType;
    if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(bufferType)) {
        const auto origDistAttr = distributedType.getDistribution();
        VPUX_THROW_UNLESS(VPU::isDuplicated(origDistAttr), "Only support DUPLICATED distributed buffer type");

        // When DistributionInfoAttr has explicit per cluster memory/compute shapes, recompute them for the new shape
        // Since changeShape is not appliable for explicit distribution
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistAttr)) {
            auto ctx = bufferType.getContext();
            auto duplicatedOutputMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
            auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                    newShape, duplicatedOutputMode, nullptr, origDistAttr.getNumClusters(), nullptr,
                    origDistAttr.getUniformDistributedSegments(), bufferType.getElementType(), ctx);

            auto newElemType = bufferType.getElementType();
            if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(bufferType.getElementType())) {
                newElemType = tileScalesAndZP(qType, newShape, newOffset);
            }

            auto order = mlir::AffineMapAttr::get(bufferType.getDimsOrder().toAffineMap(ctx));
            auto memSpace = mlir::cast<vpux::VPUIP::DistributedBufferType>(bufferType).getMemSpace();

            newType = VPUIP::DistributedBufferType::get(ctx, newShape.raw(), newElemType, order, memSpace,
                                                        newDistribution);

            return VPUIP::tileTypeSparsityCompression(newType, newOffset, newShape);
        }
    }

    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(bufferType.getElementType())) {
        auto newElemType = tileScalesAndZP(qType, newShape, newOffset);
        newType = bufferType.changeShapeElemType(newShape, newElemType);
    } else {
        newType = bufferType.changeShape(newShape);
    }
    return VPUIP::tileTypeSparsityCompression(newType, newOffset, newShape);
}

// Replace single allocation with 2 separate allocations. These allocations cover same memory range, but first points to
// the beginning of buffer, second points to the middle or place with offset
BuffersPair getReplacementBuffers(mlir::Value originalBuffer, vpux::Dim tileDim, mlir::OpBuilder builder) {
    const auto bufferType = mlir::cast<vpux::NDTypeInterface>(originalBuffer.getType());
    auto bufferOp = originalBuffer.getDefiningOp<VPURT::DeclareBufferOp>();

    auto origStrides = bufferType.getStrides();
    builder.setInsertionPoint(bufferOp);
    const auto getTiledBuf = [&](int64_t newOffset, int64_t newDimSize, int64_t extraOffset,
                                 StringRef locSuffix) -> mlir::Value {
        auto newType = getNewBufferType(bufferType, tileDim, newOffset, newDimSize);
        newType = newType.changeStrides(origStrides);

        auto newBufferOffset = bufferOp.getByteOffset() + extraOffset;
        const auto newLoc = takeOpLoc(bufferOp, locSuffix);

        return builder
                .create<VPURT::DeclareBufferOp>(newLoc, newType, bufferOp.getSectionAttr(),
                                                bufferOp.getSectionIndexAttr(), getIntAttr(builder, newBufferOffset),
                                                bufferOp.getSwizzlingKeyAttr())
                ->getResult(0);
    };

    const auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(bufferType, tileDim);
    const auto extraOffset = Byte(firstPartSize * origStrides[tileDim]).count();

    auto firstBuff = getTiledBuf(/*dimOffset=*/0, firstPartSize, /*byteOffset=*/0, "first_part");
    auto secondBuff = getTiledBuf(firstPartSize, secondPartSize, extraOffset, "second_part");

    return {firstBuff, secondBuff};
}

BuffersPair getConstantParts(mlir::Value originalConstant, vpux::Dim tileDim, mlir::OpBuilder builder) {
    auto cstOp = originalConstant.getDefiningOp<Const::DeclareOp>();

    const auto cstType = mlir::cast<vpux::NDTypeInterface>(cstOp.getOutput().getType());
    const auto origShape = cstType.getShape();
    builder.setInsertionPoint(cstOp);
    const auto createCstPart = [&](int64_t tileOffset, int64_t newDimSize, StringRef locSuffix) -> mlir::Value {
        Shape offset(SmallVector<int64_t>(origShape.size(), 0));
        offset[tileDim] = tileOffset;

        const auto newShape = getSplitShape(cstType, tileDim, newDimSize);
        const auto newLoc = takeOpLoc(cstOp, locSuffix);
        return builder.createOrFold<VPUIP::SubViewOp>(newLoc, cstOp, offset.raw(), newShape.raw());
    };

    const auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(cstType, tileDim);
    auto firstCst = createCstPart(0, firstPartSize, "first_part");
    auto secondCst = createCstPart(firstPartSize, secondPartSize, "second_part");

    return {firstCst, secondCst};
}

void createDMATask(VPURT::TaskOp originalTaskOp, VPUIP::NNDMAOp originalDmaOp, mlir::Value input, mlir::Value output,
                   int64_t port, StringRef locSuffix, mlir::OpBuilder builder) {
    auto newLoc = takeOpLoc(originalTaskOp, locSuffix);
    auto newDmaOp = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            builder, originalTaskOp.getWaitBarriers(), originalTaskOp.getUpdateBarriers(), newLoc, input, output, port,
            originalDmaOp.getIsOutOfOrder(), originalDmaOp.getIsCritical(), originalDmaOp.getSpillIdAttr(),
            /*compress_candidate=*/false);

    if (originalDmaOp.getProfilingBufferMgmt()) {
        newDmaOp.setProfilingBufferMgmt(true);
    }
}

void createDMATask(VPURT::TaskOp originalTaskOp, VPUIP::GatherDMAOp originalGatherDmaOp, mlir::Value indices,
                   mlir::Value output, int64_t port, StringRef locSuffix, mlir::OpBuilder builder) {
    auto newLoc = takeOpLoc(originalTaskOp, locSuffix);
    VPURT::wrapIntoTaskOp<VPUIP::GatherDMAOp>(
            builder, originalTaskOp.getWaitBarriers(), originalTaskOp.getUpdateBarriers(), newLoc,
            originalGatherDmaOp.getInput(), indices, output, originalGatherDmaOp.getElementSize(),
            originalGatherDmaOp.getPadding(), port);
}

template <typename DmaOpType>
void replaceDmaWithTwoParts(VPURT::TaskOp taskOp, DmaOpType dmaOp, BuffersPair inputs, BuffersPair outputs,
                            int64_t numDmaPorts, ArrayRef<mlir::Value> removableInputs, mlir::OpBuilder builder,
                            vpux::Logger log) {
    builder.setInsertionPoint(taskOp);
    const auto firstPartPort = dmaOp.getPort().value();
    const auto secondPartPort = (firstPartPort + 1) % numDmaPorts;

    createDMATask(taskOp, dmaOp, inputs.first, outputs.first, firstPartPort, "first_part", builder);
    createDMATask(taskOp, dmaOp, inputs.second, outputs.second, secondPartPort, "second_part", builder);

    taskOp->erase();
    for (mlir::Value input : removableInputs) {
        if (auto* op = input.getDefiningOp()) {
            if (op->use_empty()) {
                op->erase();
            }
        }
    }

    log.trace("Replaced DMA with two parts");
}

// Trivial constant is constant without LAST or PREFERRED_LAST transformation, so SubView transformation can be
// attached to the end of list
bool isTrivialConst(Const::DeclareOp cstOp) {
    const auto& contentAttr = cstOp.getContentAttr();
    auto transformations = contentAttr.getTransformations();
    return transformations.empty() ||
           transformations.back().getPositionRequirement() == vpux::Const::details::PositionRequirement::NONE;
}

// Non trivial transforms requires folding and flattening to keep content correct
void splitFoldedConstToBufferDma(VPURT::TaskOp taskOp, VPUIP::NNDMAOp dmaOp, Const::DeclareOp cstOp, vpux::Dim tileDim,
                                 int64_t numDmaPorts, mlir::OpBuilder builder, vpux::Logger log) {
    const auto cstType = mlir::cast<vpux::NDTypeInterface>(cstOp.getOutput().getType());
    const auto strides = cstType.getStrides();
    // In case of subbyte type, which has non-byte stride along tiling dim - don't attempt to split this constant
    const auto tileDimStride = strides[tileDim];
    if (tileDimStride.count() % CHAR_BIT != 0) {
        log.trace("Can't split constant with non-byte stride");
        return;
    }
    auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(cstType, tileDim);
    if (firstPartSize != secondPartSize) {
        log.trace("Can't split constant with odd tiling dim");
        return;
    }

    const auto content = cstOp.getContent();
    const auto contentType = content.getType();
    const auto contentElemType = contentType.getElementType();
    const auto elemTypeBitSize = contentType.getElemTypeSize().count();
    const auto isUnsupportedSubByteStorageType = elemTypeBitSize < CHAR_BIT && elemTypeBitSize > 1;
    if (isUnsupportedSubByteStorageType) {
        log.trace("Can't split constant with unsupported element type");
        return;
    }
    log.trace("Splitting FoldedConst->Buffer DMA");
    const auto bufSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
    std::vector<char> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(tempBuf.data(), bufSize));

    auto rankedTensorType = mlir::cast<mlir::RankedTensorType>(contentType);
    if (auto qtype = mlir::dyn_cast<mlir::quant::QuantizedType>(contentElemType)) {
        rankedTensorType =
                mlir::cast<mlir::RankedTensorType>(contentType.changeElemType(normalizeQuantStorageType(qtype)));
    }

    const auto rankedElemType = rankedTensorType.getElementType();
    const auto fullShape = rankedTensorType.getShape();
    SmallVector<int64_t> newShapeVec(fullShape.begin(), fullShape.end());
    auto actualTileDims = getNonOneDim(ShapeRef(newShapeVec));
    if (actualTileDims.size() == 0) {
        log.trace("Can't split constant with all ones shape");
        return;
    }
    newShapeVec[actualTileDims[0].ind()] /= 2;

    builder.setInsertionPoint(cstOp);
    const auto createCstPart = [&](int64_t tileOffset, int64_t newDimSize, StringRef locSuffix) -> mlir::Value {
        const size_t offsetSize = Byte(tileOffset * tileDimStride).count();
        const size_t bufferSize = Byte(newDimSize * tileDimStride).count();
        char* baseContentPtr = tempBuf.data() + offsetSize;
        ArrayRef<char> partContent(baseContentPtr, baseContentPtr + bufferSize);
        const auto partRankedTensorType = rankedTensorType.clone(newShapeVec, rankedElemType);
        const auto denseAttr = Const::createConstContent(partRankedTensorType, partContent);
        const auto newLoc = takeOpLoc(cstOp, locSuffix);
        const auto newType = getNewBufferType(cstType, tileDim, tileOffset, newDimSize);

        return builder.create<Const::DeclareOp>(newLoc, newType, Const::ContentAttr::get(denseAttr));
    };

    auto firstCst = createCstPart(0, firstPartSize, "first_part");
    auto secondCst = createCstPart(firstPartSize, secondPartSize, "second_part");

    BuffersPair inputBuffers = {firstCst, secondCst};
    auto outputBuffers = getReplacementBuffers(dmaOp.getOutputBuff(), tileDim, builder);
    SmallVector<mlir::Value> removableInputs;
    removableInputs.push_back(dmaOp.getInput());
    removableInputs.push_back(dmaOp.getOutputBuff());

    replaceDmaWithTwoParts(taskOp, dmaOp, inputBuffers, outputBuffers, numDmaPorts, removableInputs, builder, log);
}

template <typename DmaOpType>
void splitDMATask(VPURT::TaskOp taskOp, DmaOpType dmaOp, int64_t numDmaPorts, mlir::Value origInput,
                  std::function<std::optional<vpux::Dim>(DmaOpType, vpux::Logger)> getTilingDim,
                  std::function<BuffersPair(mlir::Value, vpux::Dim, mlir::OpBuilder)> getReplacementInputBuffers,
                  std::function<bool(DmaOpType, vpux::Dim, vpux::Logger)> alignmentCheck, mlir::OpBuilder builder,
                  vpux::Logger log) {
    auto maybeTileDim = getTilingDim(dmaOp, log);
    if (!maybeTileDim.has_value()) {
        log.trace("No valid tiling dimension found, skip");
        return;
    }
    const auto tileDim = maybeTileDim.value();

    const auto outBuffType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getOutputBuff().getType());
    const auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(outBuffType, tileDim);
    if (firstPartSize == 0 || secondPartSize == 0) {
        log.trace("Split would create empty part (first={0}, second={1}), skip", firstPartSize, secondPartSize);
        return;
    }

    if (alignmentCheck && !alignmentCheck(dmaOp, tileDim, log)) {
        log.trace("Split would create unaligned input, skip");
        return;
    }

    // Generate input and output buffers
    auto inputBuffers = getReplacementInputBuffers(origInput, tileDim, builder);
    auto outputBuffers = getReplacementBuffers(dmaOp.getOutputBuff(), tileDim, builder);

    SmallVector<mlir::Value> removableInputs = {origInput, dmaOp.getOutputBuff()};
    replaceDmaWithTwoParts(taskOp, dmaOp, inputBuffers, outputBuffers, numDmaPorts, removableInputs, builder, log);
}

std::optional<vpux::Dim> getNNDMATilingDim(VPUIP::NNDMAOp dmaOp, vpux::Logger /*log*/) {
    return VPUIP::getCopyDMATilingDim(dmaOp);
}

std::optional<vpux::Dim> getGatherDMATilingDim(VPUIP::GatherDMAOp gatherOp, vpux::Logger /*log*/) {
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(gatherOp.getIndices().getType());
    return vpux::getHighestNonTrivialDim(indicesType.getShape(), indicesType.getDimsOrder());
}

bool checkGatherIndicesAlignment(VPUIP::GatherDMAOp gatherOp, vpux::Dim tileDim, vpux::Logger log) {
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(gatherOp.getIndices().getType());
    const int64_t indicesTileDimSize = indicesType.getShape()[tileDim];
    if (indicesTileDimSize % vpux::VPU::INDICES_ALIGNMENT != 0) {
        log.trace("Split would create unaligned indices (size={0}, alignment={1}), skip", indicesTileDimSize,
                  vpux::VPU::INDICES_ALIGNMENT);
        return false;
    }
    return true;
}

void handleNNDMASplit(VPURT::TaskOp taskOp, VPUIP::NNDMAOp dmaOp, int64_t numDmaPorts, mlir::OpBuilder builder,
                      vpux::Logger log) {
    if (auto inputCst = dmaOp.getInput().getDefiningOp<Const::DeclareOp>()) {
        if (!isTrivialConst(inputCst)) {
            const auto maybeTileDim = VPUIP::getCopyDMATilingDim(dmaOp);
            if (maybeTileDim.has_value()) {
                splitFoldedConstToBufferDma(taskOp, dmaOp, inputCst, maybeTileDim.value(), numDmaPorts, builder, log);
            }
            return;
        }
    }

    auto getReplacementInputBuffers = [&](mlir::Value input, vpux::Dim tileDim,
                                          mlir::OpBuilder builder) -> BuffersPair {
        if (auto cstOp = input.getDefiningOp<Const::DeclareOp>()) {
            return getConstantParts(input, tileDim, builder);
        } else {
            return getReplacementBuffers(input, tileDim, builder);
        }
    };

    splitDMATask<VPUIP::NNDMAOp>(taskOp, dmaOp, numDmaPorts, dmaOp.getInput(), getNNDMATilingDim,
                                 getReplacementInputBuffers, nullptr, builder, log);
}

void handleGatherDMASplit(VPURT::TaskOp taskOp, VPUIP::GatherDMAOp gatherDMAOp, int64_t numDmaPorts,
                          mlir::OpBuilder builder, vpux::Logger log) {
    splitDMATask<VPUIP::GatherDMAOp>(taskOp, gatherDMAOp, numDmaPorts, gatherDMAOp.getIndices(), getGatherDMATilingDim,
                                     getReplacementBuffers, checkGatherIndicesAlignment, builder, log);
}

//
// SplitDMAToBalanceLoad
//

class SplitDMAToBalanceLoad final : public VPUIP::impl::SplitDMAToBalanceLoadBase<SplitDMAToBalanceLoad> {
public:
    explicit SplitDMAToBalanceLoad(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitDMAToBalanceLoad::safeRunOnFunc() {
    auto func = getOperation();
    auto dmaPortCount = config::getNumOfDMAPorts(func);
    if (dmaPortCount != 2) {
        return;
    }

    func->walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() != config::ExecutorKind::DMA_NN) {
            return;
        }
        mlir::OpBuilder builder(taskOp.getOperation());

        if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp())) {
            VPUX_THROW_UNLESS(dmaOp.getPort().has_value(), "DMA at '{0}' has no portId", dmaOp->getLoc());

            if (!dmaOp.getSplitCandidate()) {
                return;
            }
            _log.trace("Found split candidate at '{0}'", dmaOp->getLoc());
            mlir::Operation* inputOp = dmaOp.getInput().getDefiningOp();
            if (!mlir::isa<VPURT::DeclareBufferOp, Const::DeclareOp>(inputOp)) {
                _log.warning("Can't split op because of unsupported source");
                return;
            }
            handleNNDMASplit(taskOp, dmaOp, dmaPortCount, builder, _log.nest());
            return;
        }

        if (auto gatherDMAOp = mlir::dyn_cast<VPUIP::GatherDMAOp>(taskOp.getInnerTaskOp())) {
            VPUX_THROW_UNLESS(gatherDMAOp.getPort().has_value(), "Gather DMA at '{0}' has no portId",
                              gatherDMAOp->getLoc());
            if (!gatherDMAOp.getSplitCandidate()) {
                return;
            }
            _log.trace("Found split candidate at '{0}'", gatherDMAOp->getLoc());
            handleGatherDMASplit(taskOp, gatherDMAOp, dmaPortCount, builder, _log.nest());
            return;
        }
    });
    _log.trace("Done");
}

}  // namespace

//
// createSplitDMAToBalanceLoadPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSplitDMAToBalanceLoadPass(Logger log) {
    return std::make_unique<SplitDMAToBalanceLoad>(log);
}
