//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/utils/permute_dma.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/utils/permute_dma.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"

namespace vpux::arch40xx {

mlir::AffineMap getLogicalTransposeFromMemPermute(NDTypeInterface inType, NDTypeInterface outType,
                                                  mlir::AffineMap memPermute) {
    VPUX_THROW_WHEN(inType.getRank() != outType.getRank(), "Rank mismatch between input type and output type");

    auto ctx = inType.getContext();
    const auto mappingInMemToLogical = VPUIP::getSmallVectorFromAffineMap(inType.getDimsOrder().toAffineMap(ctx));
    const auto mappingOutMemToLogical = VPUIP::getSmallVectorFromAffineMap(outType.getDimsOrder().toAffineMap(ctx));
    const auto mappingOutToInMem = VPUIP::getSmallVectorFromAffineMap(memPermute);

    auto mappingOutToInLogical = mlir::SmallVector<int64_t>(inType.getRank());

    for (auto index : irange(inType.getRank())) {
        mappingOutToInLogical[mappingOutMemToLogical[index]] = mappingInMemToLogical[mappingOutToInMem[index]];
    }

    return mlir::AffineMap::getPermutationMap(mappingOutToInLogical, ctx);
}

mlir::LogicalResult UnrollSingleClusterPermuteDMA::unroll(VPUIP::PermuteDMAOp permuteOp,
                                                          mlir::PatternRewriter& rewriter, int64_t portCount, Logger) {
    VPUX_THROW_WHEN(permuteOp.getInternalDataFlowAttr(), "Already unrolled");
    VPUX_THROW_WHEN(portCount < 1, "Invalid number of ports (expected at least 1, but got {0})", portCount);

    auto ctx = permuteOp->getContext();

    auto origTaskOp = permuteOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(origTaskOp != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(origTaskOp);

    auto origInType = mlir::cast<NDTypeInterface>(permuteOp.getInput().getType());
    auto origInBuffer = permuteOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto origInBufferOffset = origInBuffer.getByteOffset();

    auto origOutType = mlir::cast<NDTypeInterface>(permuteOp.getOutput().getType());
    auto origOutBuffer = permuteOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto origOutBufferOffset = origOutBuffer.getByteOffset();

    // Super paranoid check
    VPUX_THROW_WHEN(origInType.getRank() != origOutType.getRank(), "Rank mismatch between input type and output type");

    // mem_perm attribute maps out memory dims to in memory dims.
    //
    // The modes of operation of PermuteDMA could be summarized as follows:
    //  - logical transpose -> logical shape changes, but the memory layout remains unchanged
    //      1x2x3x4 NHWC -> 1x3x2x4 NHWC
    //  - memory layout change -> logical shape is unchanged, but the memory layout is different
    //      1x2x3x4 NHWC to 1x2x3x4 NCHW
    //  - logical transpose + memory layout change -> both logical shape and memory layout are different
    //      1x2x3x4 NHWC -> 1x3x2x4 NCHW
    //
    // To make things more explicit:
    //  - memory layout change is represented through the memory layout of the input and output types
    //  - logical transpose is represented through mappingOrder (AffineMap) which maps output logical dims to input
    //  logical dims
    auto mappingOrder =
            getLogicalTransposeFromMemPermute(origInType, origOutType, permuteOp.getMemPermAttr().getAffineMap());
    auto mappingOutToInLogical = VPUIP::getSmallVectorFromAffineMap(mappingOrder);
    auto mappingInToOutLogical = VPUIP::getSmallVectorFromAffineMap(mlir::inversePermutation(mappingOrder));

    auto workingInShape = Shape(origInType.getShape().raw());
    auto origInStrides = origInType.getStrides();

    auto workingOutShape = Shape(origOutType.getShape().raw());
    auto origOutStrides = origOutType.getStrides();

    // Temporary solution to treat case where Expand is fused with Permute, which results in output size > input size
    // (see E#173193). Use input dim sizes and keep output strides.
    for (auto index : irange(origOutType.getRank())) {
        workingOutShape[Dim(index)] = workingInShape[Dim(mappingOutToInLogical[index])];
    }

    // Initialize properly here indexes and task count in case no splittable dim is found
    int64_t inSplitDimIndex = 0;
    int64_t outSplitDimIndex = mappingInToOutLogical[inSplitDimIndex];
    int64_t newTaskCount = 1;

    auto hasPortAssigned = permuteOp.getPort().has_value();

    // Initialize new port
    // All cluster tasks will have port assigned
    // In case of single cluster task, if splitting to all ports is not possible, always use port 0
    int64_t newPort = 0;

    // Split only byte-aligned element types
    if (origInType.getElemTypeSize() % Byte(1).to<Bit>() == 0) {
        // Find a split candidate (search in mem order to find largest continuous chunk)
        for (auto index : VPUIP::getSmallVectorFromAffineMap(origInType.getDimsOrder().toAffineMap(ctx))) {
            // Find the first non-trivial dim (i.e. dim size > 1) that is evenly divided by number of ports
            // This is needed to ensure load balancing, particularly when reading from DDR
            if (workingInShape[Dim(index)] > 1 && workingInShape[Dim(index)] % portCount == 0) {
                inSplitDimIndex = index;
                newTaskCount = portCount;
                outSplitDimIndex = mappingInToOutLogical[inSplitDimIndex];
                break;
            }
        }
    }

    auto origSplitDimSize = workingInShape[Dim(inSplitDimIndex)];
    auto initialSplitDimSize = workingInShape[Dim(inSplitDimIndex)] / newTaskCount;
    auto currentSplitDimSize = initialSplitDimSize;

    // Offsets needed for per-axis quant type updates
    auto workingInShapeOffsets = Shape(workingInShape.size());
    auto workingOutShapeOffsets = Shape(workingOutShape.size());

    const auto getNewElementType = [](NDTypeInterface origType, ShapeRef newShape, ShapeRef newOffset) {
        auto elemType = origType.getElementType();
        if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
            // tileScalesAndZP will update type only when we actually tiled over quantized axis
            elemType = tileScalesAndZP(qType, newShape, newOffset);
        }

        return elemType;
    };

    const auto createNewBuffer = [](mlir::PatternRewriter& rewriter, VPURT::TaskOp taskOp,
                                    VPURT::DeclareBufferOp existingBuffer, NDTypeInterface newType, int64_t newOffset) {
        if (newType.getMemSpace().getIndex().has_value()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, existingBuffer, taskOp.getLoc(), newType,
                                                           existingBuffer.getSection(),
                                                           newType.getMemSpace().getIndex().value(), newOffset);
        } else {
            if (existingBuffer.getSectionIndex().has_value()) {
                return VPURT::createOp<VPURT::DeclareBufferOp>(
                        rewriter, existingBuffer, taskOp.getLoc(), newType, existingBuffer.getSection(),
                        parseIntArrayAttr<int64_t>(existingBuffer.getSectionIndex().value()), newOffset);
            }

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, existingBuffer, taskOp.getLoc(), newType,
                                                           existingBuffer.getSection(), newOffset);
        }
    };

    const auto getNewInType = [&getNewElementType](NDTypeInterface origType, ShapeRef newShape, ShapeRef newOffsets,
                                                   StridesRef newStrides) -> NDTypeInterface {
        auto newElementType = getNewElementType(origType, newShape, newOffsets);

        VPUX_THROW_UNLESS(mlir::isa_and_nonnull<mlir::MemRefType>(origType), "Unexpected input type");

        return getMemRefType(newShape, newElementType, origType.getDimsOrder(), origType.getMemSpace(), newStrides);
    };

    const auto getNewOutType = [&getNewElementType](mlir::MLIRContext* ctx, NDTypeInterface origType, ShapeRef newShape,
                                                    ShapeRef newOffsets, StridesRef newStrides) -> NDTypeInterface {
        auto newElementType = getNewElementType(origType, newShape, newOffsets);

        if (auto dstDistributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(origType)) {
            auto distributionAttr = dstDistributedType.getDistribution();
            VPUX_THROW_WHEN(
                    distributionAttr.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "Issues with unrolling PermuteNNDMA; Buffer has distributed type != DUPLICATED after unroll");

            if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributionAttr)) {
                distributionAttr = VPU::getNonOverlappedDistributedAttr(
                        newShape, distributionAttr.getMode(), nullptr, distributionAttr.getNumClusters(), nullptr,
                        distributionAttr.getUniformDistributedSegments(), ctx);
            }

            // We now convert to identity logical to memory mapping
            const auto layout = mlir::AffineMapAttr::get(origType.getDimsOrder().toAffineMap(ctx));
            return VPUIP::DistributedBufferType::get(ctx, newShape, newElementType, layout, origType.getMemSpace(),
                                                     distributionAttr);
        }

        return getMemRefType(newShape, newElementType, origType.getDimsOrder(), origType.getMemSpace(), newStrides);
    };

    // Initialize variables here to allow loop to handle single iteration cases
    // We update the buffers and types since we are switching to identity logical to memory mapping
    auto newInType = getNewInType(origInType, workingInShape, workingInShapeOffsets, origInStrides);
    auto newInBuffer = createNewBuffer(rewriter, origTaskOp, origInBuffer, newInType, origInBufferOffset);

    auto newOutType = getNewOutType(ctx, origOutType, workingOutShape, workingOutShapeOffsets, origOutStrides);
    auto newOutBuffer = createNewBuffer(rewriter, origTaskOp, origOutBuffer, newOutType, origOutBufferOffset);

    for (auto index : irange(newTaskCount)) {
        if (newTaskCount > 1) {
            if (index == newTaskCount - 1) {
                // For last iter use remaining size for cases where dim size is not divisible nicely
                currentSplitDimSize = origSplitDimSize - initialSplitDimSize * index;
            }
            newPort = index;

            // Compute new shapes
            workingInShape[Dim(inSplitDimIndex)] = currentSplitDimSize;
            workingOutShape[Dim(outSplitDimIndex)] = currentSplitDimSize;

            // Compute new offsets.
            // For simplicity, the splitting interleaves accesses to the original shapes.
            // Pretty heavy assumption here that strides will turn out to be byte aligned.
            // Jump only over elements in the dimension we split.
            // For a compact shape, if we split over the highest order dim, the access will be continuous.
            auto newInBufferOffset =
                    initialSplitDimSize * origInStrides[Dim(inSplitDimIndex)].to<Byte>().count() * index +
                    origInBufferOffset;
            auto newOutBufferOffset =
                    initialSplitDimSize * origOutStrides[Dim(outSplitDimIndex)].to<Byte>().count() * index +
                    origOutBufferOffset;

            workingInShapeOffsets[Dim(inSplitDimIndex)] = initialSplitDimSize * index;
            newInType = getNewInType(origInType, workingInShape, workingInShapeOffsets, origInStrides);
            newInBuffer = createNewBuffer(rewriter, origTaskOp, newInBuffer, newInType, newInBufferOffset);

            workingOutShapeOffsets[Dim(outSplitDimIndex)] = initialSplitDimSize * index;
            newOutType = getNewOutType(ctx, origOutType, workingOutShape, workingOutShapeOffsets, origOutStrides);
            newOutBuffer = createNewBuffer(rewriter, origTaskOp, newOutBuffer, newOutType, newOutBufferOffset);
        }

        auto loopOrder =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(VPUIP::getLinearMemOrder(newInType), ctx));

        auto internalDataFlowAttr = VPUIP::InternalDataFlowAttr::get(ctx, newInType, newOutType,
                                                                     mlir::AffineMapAttr::get(mappingOrder), loopOrder);

        const auto newLoc = appendLoc(origTaskOp->getLoc(), "_unrolled_permuteDMA");

        // Override port if no splitting can be done and port was already assigned by cluster unrolling
        if (hasPortAssigned && newTaskCount == 1) {
            newPort = permuteOp.getPort().value();
        }

        VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, origTaskOp.getWaitBarriers(), origTaskOp.getUpdateBarriers(), newLoc, newInBuffer,
                newOutBuffer, getIntAttr(rewriter, newPort), permuteOp.getIsOutOfOrderAttr(),
                permuteOp.getIsCriticalAttr(),
                /*mem_perm*/ nullptr, /* dma_descriptor */ nullptr, permuteOp.getDmaHwpIdAttr(),
                permuteOp.getProfilingMetadataAttr(), internalDataFlowAttr);
    }

    rewriter.eraseOp(origTaskOp);
    return mlir::success();
}

mlir::LogicalResult rewritePermuteDMA(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter, int64_t portCount,
                                      Logger logger) {
    return arch37xx::unrollPermuteDMA<arch40xx::UnrollSingleClusterPermuteDMA, arch37xx::UnrollMultiClusterPermuteDMA>(
            permuteOp, rewriter, portCount, logger);
}

}  // namespace vpux::arch40xx
