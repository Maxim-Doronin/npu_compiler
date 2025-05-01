
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_WRAPWITHPERMUTEASNNDMA
#define GEN_PASS_DEF_WRAPWITHPERMUTEASNNDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset) {
    const auto elemType = originType.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        return originType.changeShapeElemType(shape, newQType);
    }

    return originType.changeShape(shape);
}

bool isSplitContinuousBufferType(VPUIP::DistributedBufferType distributedType) {
    auto isCompactType = [](vpux::NDTypeInterface origType) {
        const auto shape = origType.getShape();
        const auto strideReqs = StrideReqs::compact(shape.size());
        return strideReqs.checkStrides(origType);
    };

    auto innerType = distributedType.getCompactType();

    if (!isMemoryContiguousWithTiling(distributedType)) {
        return false;
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto tileCount = distributionAttr.getNumClusters().getInt();
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    const auto tileInnerType = [&](vpux::NDTypeInterface innerType) {
        SmallVector<vpux::NDTypeInterface> newTypes(tileCount);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(innerType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        }

        return newTypes;
    };
    auto outTypes = tileInnerType(innerType);
    return llvm::all_of(outTypes, isCompactType);
}

VPUIP::DistributedBufferType createDMADistributedTensorType(mlir::MLIRContext* ctx, vpux::NDTypeInterface operandType,
                                                            mlir::IntegerAttr tileCount, VPU::ArchKind arch,
                                                            bool uniformDistributedSegments) {
    const auto distMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1, tileCount.getInt(), 1});
    const auto isSparse = mlir::isa<vpux::VPUIP::SparseBufferType>(operandType);
    const auto heightAlignment =
            VPU::getSOHMinimalHeightAlignment(operandType.getShape(), tileCount.getInt(), isSparse, arch);
    const auto alignment =
            heightAlignment == 1 ? nullptr : getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1, heightAlignment, 1});
    const auto uniformDistributedSegmentsAttr = uniformDistributedSegments ? mlir::UnitAttr::get(ctx) : nullptr;
    const auto distributionAttr =
            VPU::DistributionInfoAttr::get(ctx, distMode, numTiles, nullptr, nullptr, nullptr, tileCount, alignment,
                                           uniformDistributedSegmentsAttr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto order = mlir::AffineMapAttr::get(operandType.getDimsOrder().toAffineMap(ctx));
    auto elemType = operandType.getElementType();

    return VPUIP::DistributedBufferType::get(ctx, operandType.getShape().raw(), elemType, order, memSpace,
                                             distributionAttr);
}

SmallVector<mlir::Operation*> getPureViewLikeOpChains(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op->hasOneUse(), "Op has more than one uses at '{0}'", op->getLoc());
    SmallVector<mlir::Operation*> viewLikeOps;
    auto user = *op->getUsers().begin();
    while (user->hasOneUse()) {
        if (!mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp, VPUIP::ShapeCastOp>(user) && user->hasOneUse()) {
            break;
        }
        viewLikeOps.push_back(user);
        user = *user->getUsers().begin();
    }
    return viewLikeOps;
}

// check pattern: Copy(ddr->cmx) -> sw.kernel(memPermute)
bool checkPermuteWithoutCopyBackPattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!VPUIP::isMemPermSwKernel(swKernelOp)) {
        return false;
    }

    log.trace("Got MemPermute SwKernel at {0}. Try to find fuse pattern.", swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, log)) {
        log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getOperand(0).getType());
    const auto memPerm = getMemPermFromSwKernel(swKernelOp).value();
    if (VPUIP::isSplitNeededForPermuteDMA(inputType, memPerm)) {
        log.trace("PermuteDMA split is not supported for fuse MemPermute with copy pattern.");
        return false;
    }

    if (!swKernelOp->hasOneUse()) {
        log.nest().trace("VPUIP.SwKernel has more than one use at {0}", swKernelOp->getLoc());
        return false;
    }

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (copyInCMXOp == nullptr || !copyInCMXOp->hasOneUse() || vpux::VPUIP::hasDistributedOperand(copyInCMXOp)) {
        return false;
    }

    const auto copyInInputType = mlir::cast<vpux::NDTypeInterface>(copyInCMXOp.getInput().getType());
    const auto copyInOutputType = mlir::cast<vpux::NDTypeInterface>(copyInCMXOp.getOutput().getType());
    return copyInInputType.getMemoryKind() == VPU::MemoryKind::DDR &&
           copyInOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
}

// check pattern: Copy(ddr->cmx) -> sw.kernel(memPermute) -> Copy (cmx-> ddr)
bool checkPermutePattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!checkPermuteWithoutCopyBackPattern(swKernelOp, log)) {
        return false;
    }
    if (!swKernelOp->hasOneUse()) {
        log.nest().trace("VPUIP.SwKernel has more than one use at {0}", swKernelOp->getLoc());
        return false;
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (copyBackToDDROp == nullptr || !copyBackToDDROp->hasOneUse() ||
        vpux::VPUIP::hasDistributedOperand(copyBackToDDROp)) {
        return false;
    }

    const auto copyBackInputType = mlir::cast<vpux::NDTypeInterface>(copyBackToDDROp.getInput().getType());
    const auto copyBackOutputType = mlir::cast<vpux::NDTypeInterface>(copyBackToDDROp.getOutput().getType());
    return copyBackInputType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
           copyBackOutputType.getMemoryKind() == VPU::MemoryKind::DDR;
}

// Check pattern 1: Copy(ddr->cmx) -> sw.kernel(memPermute) -> Copy (cmx-> ddr) -> Copy (ddr-> cmx)
// Check pattern 2: Copy(ddr->cmx) -> sw.kernel(memPermute) -> Copy (cmx-> ddr) -> ClusterCopy (ddr-> cmx)
bool checkPermuteWithCopyPattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!checkPermutePattern(swKernelOp, log)) {
        return false;
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    VPUX_THROW_WHEN(vpux::VPUIP::hasDistributedOperand(copyBackToDDROp), "Copy back to DDR can't be distributed");

    auto copyToNCEOp = *(copyBackToDDROp->getUsers().begin());
    if (vpux::VPUIP::hasDistributedOperand(copyToNCEOp)) {
        // It is difficult to use the general method to fuse Permute the with next Cluster Copy Op
        // which has the stride. For example, activation with NHWC layout, need tile at Channel.
        // It is necessary to check the split buffer is continuous.
        const auto clusterOutput = VPUIP::getLayerOutputs(copyToNCEOp)[0];
        const auto clusterOutputType = mlir::cast<vpux::NDTypeInterface>(clusterOutput.getType());

        if (!isSplitContinuousBufferType(mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(clusterOutputType))) {
            return false;
        }
        auto permuteInType = mlir::dyn_cast<vpux::NDTypeInterface>(swKernelOp.getOperand(0).getType());
        auto permuteOutType = mlir::dyn_cast<vpux::NDTypeInterface>(swKernelOp.getResult(0).getType());

        auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).value();
        if (memPerm == DimsOrder::NWHC.toAffineMap(swKernelOp->getContext())) {
            log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
            return false;
        }

        auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
        const auto dmaPortNum = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

        auto dmaSubShapes = VPUIP::getPermuteDMASubInputShapes(VPU::getArch(swKernelOp), permuteInType, permuteOutType,
                                                               memPerm, dmaPortNum, log);
        // If fuse Permute with next Cluster Copy Op and PermuteDMA need unroll to severl Sub DMA tasks,
        // Find a scenerior has regression. Need investigate the root cause and find a cost model for that.
        // For example: Shape size with 1x4420x1x2, mode is DUPLICATED.
        // It will be unrolled to 18 PermuteDMA with shape size 1x256x1x2 (17) + 1x68x1x2 (1)
        if (!dmaSubShapes.has_value() || dmaSubShapes.value().size() > 2) {
            return false;
        }

        if (!VPUIP::doesPermuteDMATileDimSupportWrapInCluster(
                    permuteInType, permuteOutType, memPerm,
                    mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(clusterOutputType), log)) {
            return false;
        }
    }

    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(copyToNCEOp);
    if (childCopyOp == nullptr) {
        return false;
    }

    const auto childOutputType = mlir::cast<vpux::NDTypeInterface>(childCopyOp.getOutput().getType());
    return childOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
}

// Check pattern: TilingCopy(CMX->DDR) -> Copy(DDR->CMX) -> sw.kernel(memPermute)
bool checkTilingCopyWithPermutePattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    log.trace("Got sw kernel op at {0}. Try to find permute pattern.", swKernelOp->getLoc());
    if (!checkPermuteWithoutCopyBackPattern(swKernelOp, log)) {
        return false;
    }

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    VPUX_THROW_WHEN(vpux::VPUIP::hasDistributedOperand(copyInCMXOp), "Copy in CMX can't be distributed");
    auto copyInput = copyInCMXOp->getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (copyInput == nullptr || !vpux::VPUIP::hasDistributedOperand(copyInput) || !copyInput->hasOneUse()) {
        return false;
    }

    auto inDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(copyInput->getOperand(0).getType());
    if (inDistributedType == nullptr) {
        return false;
    }

    const auto inReqs = StrideReqs::compact(inDistributedType.getRank());
    if (!inReqs.checkStrides(inDistributedType)) {
        log.trace("Skip complex case: input is strided");
        return false;
    }

    auto inMode = inDistributedType.getDistribution().getMode().getValue();
    return VPU::bitEnumContainsAny(inMode, VPU::DistributionMode::DUPLICATED);
}

bool onlyExpandAtChannel(VPUIP::ExpandOp expandOp) {
    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());

    if (padsBegin.size() != 4 || padsEnd.size() != 4) {
        return false;
    }

    const auto padValues = zip(padsBegin, padsEnd);
    for (auto padValue : padValues | indexed) {
        if (std::get<0>(padValue.value()) != 0 ||
            (std::get<1>(padValue.value()) != 0 && Dim(padValue.index()) != Dims4D::Act::C)) {
            return false;
        }
    }

    return true;
}

bool isExpandOpWrapable(VPUIP::ExpandOp expandOp, Logger log) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(expandOp.getOutput().getType());

    if (outputType.getDimsOrder() != DimsOrder::NCHW && outputType.getDimsOrder() != DimsOrder::NHWC) {
        log.nest().trace("ExpandOp convert to DMA should have NCHW or NHWC layout.");
        return false;
    }

    const auto nonZeroAxisPredicate = [](const int64_t dim) -> bool {
        return dim > 0;
    };

    const auto hasPadAndPadAtChannel = [&](mlir::ArrayAttr pads) -> bool {
        const auto padsValue = parseIntArrayAttr<int64_t>(pads);
        const auto padAxisIter = std::find_if(padsValue.begin(), padsValue.end(), nonZeroAxisPredicate);
        if (padAxisIter != padsValue.end()) {
            const auto padAxis = std::distance(padsValue.begin(), padAxisIter);
            return padAxis == Dims4D::Act::C.ind();
        }
        return false;
    };

    // Only support Expand layer with padding at channel and padding at end
    // TODO: Padding at any axis
    const auto padBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    if (std::any_of(padBegin.begin(), padBegin.end(), nonZeroAxisPredicate)) {
        log.nest().trace("Only support Expand layer with padding at the end. But got {0}.", padBegin);
        return false;
    }

    if (!hasPadAndPadAtChannel(expandOp.getPadsEnd())) {
        log.nest().trace("Only support Expand layer with padding at channel. But got {0}.", expandOp.getPadsEnd());
        return false;
    }

    if (!expandOp->hasOneUse()) {
        return false;
    }

    auto copyOutOp = *(expandOp->getUsers().begin());
    if (vpux::VPUIP::hasDistributedOperand(copyOutOp)) {
        // It is difficult to use the general method to fuse Expand the with next Cluster Copy Op
        // which has the stride. For example, activation with NHWC layout, need tile at Channel.
        // It is necessary to check the split buffer is continuous.
        const auto clusterOutput = VPUIP::getLayerOutputs(copyOutOp)[0];
        const auto clusterOutputType = mlir::cast<VPUIP::DistributedBufferType>(clusterOutput.getType());
        if (!isSplitContinuousBufferType(clusterOutputType)) {
            return false;
        }
    }

    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(copyOutOp);
    if (childCopyOp == nullptr) {
        return false;
    }

    const auto copyOutInputType = mlir::cast<vpux::NDTypeInterface>(childCopyOp.getInput().getType());
    const auto copyOutOutputType = mlir::cast<vpux::NDTypeInterface>(childCopyOp.getOutput().getType());
    if (copyOutInputType.getMemoryKind() != VPU::MemoryKind::DDR ||
        copyOutOutputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        return false;
    }

    return true;
}

// Check pattern 1: Expand(ddr->ddr) -> Copy (ddr->cmx) (U8 precision)
// Check pattern 2: Expand(ddr->ddr) -> ClusterCopy (ddr->cmx) (U8 precision)
bool checkExpandU8Pattern(VPUIP::ExpandOp expandOp, Logger log) {
    log.trace("Got ExpandOp at {0}. Try to find fuse pattern.", expandOp->getLoc());

    /*The expandOp was inserted because align to 16 on channel dim. So the expand data is useless. We can fill any data
     * to the expand data. But for convolution, when expand with floating point precision (FP16, FP8 etc.) the expand
     * data will affect the calculation results, if we fill unnormal data like NaN/Inf. For U8 precision any value is a
     * normal data, so we can fill any data to the expand data*/
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(expandOp.getOutput().getType());
    if (const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputType.getElementType());
        qType == nullptr || !qType.getStorageType().isInteger(8)) {
        log.nest().trace("ExpandOp convert to DMA should have U8 precision.");
        return false;
    }

    return isExpandOpWrapable(expandOp, log);
}

bool checkExpandFP16Pattern(VPUIP::ExpandOp expandOp, Logger log) {
    log.trace("Got ExpandOpFP16 at {0}. Try to find fuse pattern.", expandOp->getLoc());

    const auto isCopyOpWithOneUser = [&](mlir::Operation* op) -> bool {
        if (!VPUIP::hasOneOrSameUser(op)) {
            return false;
        }

        return mlir::isa_and_nonnull<VPUIP::CopyOp>(op);
    };

    const auto isNceButNotConvOpWithOneUser = [&](mlir::Operation* op) -> bool {
        if (!VPUIP::hasOneOrSameUser(op)) {
            return false;
        }

        auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op);
        return nceTask != nullptr && nceTask.getTaskType() != VPUIP::NCETaskType::CONV;
    };

    // ExpandOp aligns channels to multiples of 16 due to HW constraints
    // ExpandDMA copies only the actively used data; the expanded portion may contain uninitialized data
    // - For operations like Pooling or GroupConvolution, computations are performed per channel
    //   Thus, any data, including uninitialized memory, can serve as the expanded data without affecting outcomes
    // - However, for Convolution operations, the expanded channels are included in computations
    //   Therefore, filling these channels with abnormal data (e.g., null values) can adversely affect the results
    //
    // Illegal Pattern: "Expand -> NceExceptConv -> Convolution -> Subview"
    //   Dirty data impacts Convolution
    // Legal Pattern: "Expand -> NceExceptConv x N -> Subview"
    //   Multiple non-conv NCE operations between Expand and Subview are supported

    auto potentialCopyOrNceOp = *expandOp->getUsers().begin();
    while (isCopyOpWithOneUser(potentialCopyOrNceOp) || isNceButNotConvOpWithOneUser(potentialCopyOrNceOp)) {
        potentialCopyOrNceOp = *potentialCopyOrNceOp->getUsers().begin();
    }

    const auto expandInput = to_small_vector(getShape(expandOp.getInput()));
    const auto expandBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());

    auto subviewOp = mlir::dyn_cast<VPUIP::SubViewOp>(potentialCopyOrNceOp);
    if (subviewOp == nullptr || subviewOp.getStaticStrides().has_value()) {
        return false;
    }

    const auto staticOffsets = parseIntArrayAttr<int64_t>(subviewOp.getStaticOffsets());
    const auto staticSizes = parseIntArrayAttr<int64_t>(subviewOp.getStaticSizes());
    if (expandBegin[Dims4D::Act::C.ind()] != staticOffsets[Dims4D::Act::C.ind()] ||
        staticSizes[Dims4D::Act::C.ind()] != expandInput[Dims4D::Act::C.ind()]) {
        return false;
    }

    return isExpandOpWrapable(expandOp, log);
}

// Check pattern 1: Expand (NCHW) -> Copy(ddr->cmx) -> sw.kernel(memPermute to NHWC) -> Copy (cmx-> ddr)
//                   -> Copy (ddr-> cmx) (U8 precision)
// Check pattern 2: Expand (NCHW) -> Copy(ddr->cmx) -> sw.kernel(memPermute to NHWC) -> Copy (cmx-> ddr)
//                   -> ClusterCopy (ddr-> cmx) (U8 precision)
bool checkExpandWithPermutePattern(VPUIP::ExpandOp expandOp, Logger log) {
    log.trace("Got ExpandOp at {0}. Try to find fuse expand and permute pattern.", expandOp->getLoc());

    // Just support Expand with layout NCHW
    const auto inOrder = DimsOrder::fromValue(expandOp.getInput());
    if (inOrder != DimsOrder::NCHW) {
        log.nest().trace("Expand With Permute Pattern should with NCHW layout. Got {0}.", inOrder);
        return false;
    }

    if (!checkExpandU8Pattern(expandOp, log)) {
        return false;
    }

    auto expandUserOp = *(expandOp->getUsers().begin());
    if (vpux::VPUIP::hasDistributedOperand(expandUserOp)) {
        return false;
    }

    auto expandCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(expandUserOp);

    if (vpux::VPUIP::hasDistributedOperand(expandCopyOp)) {
        return false;
    }
    if (expandCopyOp == nullptr || !expandOp->hasOneUse()) {
        return false;
    }

    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(expandCopyOp->getUsers().begin()));
    if (swKernelOp == nullptr || !swKernelOp->hasOneUse()) {
        return false;
    }

    // Just support Permute with input layout NCHW and output layout NHWC
    const auto permuteInOrder = DimsOrder::fromValue(swKernelOp.getInputs().front());
    const auto permuteOutOrder = DimsOrder::fromValue(swKernelOp.getOutputs().front());
    if (permuteInOrder != DimsOrder::NCHW || permuteOutOrder != DimsOrder::NHWC) {
        log.nest().trace("Just support Permute with input layout NCHW and output layout NHWC. Got {0}, {1}.",
                         permuteInOrder, permuteOutOrder);
        return false;
    }

    return checkPermuteWithCopyPattern(swKernelOp, log);
}

// Check Pattern: SW.Kernel(SpaceToDepth) -> Copy(CMX->DDR) -> Copy(DDR->CMX) -> SW.Kernel(MemPermute(Reorder))
bool checkSpaceToDepthWithPermutePattern(VPUIP::SwKernelOp s2dSwKernelOp, Logger log) {
    log.trace("Checking SpaceToDepthWithPermute pattern.");

    if (!VPUIP::isSpaceToDepthSwKernel(s2dSwKernelOp)) {
        log.nest().trace("SWKernel is not SpaceToDepth.");
        return false;
    }

    log.nest().trace("Got SpaceToDepth SwKernel '{0}' at '{1}'.", s2dSwKernelOp->getName(), s2dSwKernelOp->getLoc());

    if (!s2dSwKernelOp->hasOneUse()) {
        log.nest().trace("SpaceToDepth SwKernel should have exactly one use.");
        return false;
    }

    auto copyToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(s2dSwKernelOp->getUsers().begin()));
    if (copyToDDROp == nullptr || !copyToDDROp->hasOneUse() || vpux::VPUIP::hasDistributedOperand(copyToDDROp)) {
        log.nest().trace("No copy to DDR after SpaceToDepth or copy has not exactly one use.");
        return false;
    }

    const auto copyToDDRInType = mlir::cast<vpux::NDTypeInterface>(copyToDDROp.getInput().getType());
    const auto copyToDDROutType = mlir::cast<vpux::NDTypeInterface>(copyToDDROp.getOutput().getType());
    if (copyToDDRInType.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
        copyToDDROutType.getMemoryKind() != VPU::MemoryKind::DDR) {
        log.nest().trace("Copy after SpaceToDepth is not from CMX to DDR.");
        return false;
    }

    auto copyToCMXOp = mlir::dyn_cast<VPUIP::CopyOp>(*(copyToDDROp->getUsers().begin()));
    if (copyToCMXOp == nullptr || !copyToCMXOp->hasOneUse() || vpux::VPUIP::hasDistributedOperand(copyToCMXOp)) {
        log.nest().trace("No copy back to CMX after copy to DDR or copy has not exactly one use.");
        return false;
    }

    const auto copyToCMXInType = mlir::cast<vpux::NDTypeInterface>(copyToCMXOp.getInput().getType());
    const auto copyToCMXOutType = mlir::cast<vpux::NDTypeInterface>(copyToCMXOp.getOutput().getType());
    if (copyToCMXInType.getMemoryKind() != VPU::MemoryKind::DDR ||
        copyToCMXOutType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        log.nest().trace("Copy back to CMX is not from DDR to CMX.");
        return false;
    }

    auto permuteSwKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(copyToCMXOp->getUsers().begin()));
    if (permuteSwKernelOp == nullptr) {
        log.nest().trace("No permute found.");
        return false;
    }

    if (!VPUIP::isMemPermSwKernel(permuteSwKernelOp)) {
        log.nest().trace("SWKernel is not MemPermute.");
        return false;
    }

    log.nest().trace("Got MemPermute SWKernel '{0}' at '{1}'.", permuteSwKernelOp->getName(),
                     permuteSwKernelOp->getLoc());

    const auto permuteInOrder = DimsOrder::fromValue(permuteSwKernelOp.getInputs().front());
    const auto permuteOutOrder = DimsOrder::fromValue(permuteSwKernelOp.getOutputs().front());
    const auto permuteMemPerm = VPUIP::getMemPermFromSwKernel(permuteSwKernelOp).value();
    const auto layoutReorderMemPerm =
            getPermutationFromOrders(permuteInOrder, permuteOutOrder, permuteSwKernelOp.getContext());

    // Only if mem_perm is the same as calculated from in/out orders,
    // we can take it as an layout reorder and merge it into SpaceToDepthDMA
    if (layoutReorderMemPerm != permuteMemPerm) {
        log.nest().trace("MemPermute at '{0}' does not act as an layout reorder.", permuteSwKernelOp->getLoc());
        return false;
    }

    return true;
}

// Check pattern: sw.kernel(spaceToDepth) -> Copy (cmx-> ddr) -> ClusterCopy (ddr-> cmx)
bool checkSpaceToDepthPattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!VPUIP::isSpaceToDepthSwKernel(swKernelOp)) {
        return false;
    }

    log.trace("Got SpaceToDepth SwKernel at {0}. Try to find fuse pattern.", swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, log)) {
        log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

    auto s2dInType = mlir::dyn_cast<vpux::NDTypeInterface>(swKernelOp.getOperand(0).getType());
    auto s2dOutType = mlir::dyn_cast<vpux::NDTypeInterface>(swKernelOp.getResult(0).getType());

    if (!swKernelOp->hasOneUse()) {
        log.nest().trace("VPUIP.SwKernel has more than one use at {0}", swKernelOp->getLoc());
        return false;
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (copyBackToDDROp == nullptr || !copyBackToDDROp->hasOneUse() ||
        vpux::VPUIP::hasDistributedOperand(copyBackToDDROp)) {
        return false;
    }

    const auto copyBackInputType = mlir::cast<vpux::NDTypeInterface>(copyBackToDDROp.getInput().getType());
    const auto copyBackOutputType = mlir::cast<vpux::NDTypeInterface>(copyBackToDDROp.getOutput().getType());
    if (copyBackInputType.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
        copyBackOutputType.getMemoryKind() != VPU::MemoryKind::DDR) {
        return false;
    }

    auto user = *(copyBackToDDROp->getUsers().begin());
    if (!vpux::VPUIP::hasDistributedOperand(user)) {
        return false;
    }
    // It is difficult to use the general method to fuse Permute the with next Cluster Copy Op
    // which has the stride. For example, activation with NHWC layout, need tile at Channel.
    // It is necessary to check the split buffer is continuous.
    const auto clusterOutput = VPUIP::getLayerOutputs(user)[0];
    const auto clusterOutputType = mlir::cast<VPUIP::DistributedBufferType>(clusterOutput.getType());

    if (!isSplitContinuousBufferType(clusterOutputType)) {
        return false;
    }

    // Only supports BlocksFirst NHWC->NHWC
    auto s2dAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(s2dAttrs.has_value(), "Cannot extract attributes from SpaceToDepth SwKernel '{0}'.",
                      swKernelOp.getLoc());
    auto mode = s2dAttrs.value().first.getValue();
    auto inOrder = s2dInType.getDimsOrder();
    auto outOrder = s2dOutType.getDimsOrder();
    if (!(mode == IE::SpaceToDepthMode::BLOCKS_FIRST && inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC)) {
        return false;
    }

    auto childCopyOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(user);
    if (childCopyOp == nullptr || !vpux::VPUIP::hasDistributedOperand(childCopyOp)) {
        return false;
    }

    const auto childOutputType = mlir::cast<vpux::NDTypeInterface>(childCopyOp.getOutput().getType());
    return childOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
}

// Check pattern 1: Copy(ddr->cmx) -> sw.kernel(PerAxisTile) -> Copy (cmx-> ddr) -> Copy (ddr-> cmx)
// Check pattern 2: Copy(ddr->cmx) -> sw.kernel(PerAxisTile) -> Copy (cmx-> ddr) -> ClusterCopy (ddr-> cmx)
bool checkPerAxisTilePattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!VPUIP::isPerAxisTileSwKernel(swKernelOp)) {
        return false;
    }

    log.trace("Got PerAxisTile SwKernel at {0}. Try to find fuse pattern.", swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, log)) {
        log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

    if (!swKernelOp->hasOneUse()) {
        log.nest().trace("VPUIP.SwKernel has more than one use at {0}", swKernelOp->getLoc());
        return false;
    }

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (copyInCMXOp == nullptr || !copyInCMXOp->hasOneUse() || vpux::VPUIP::hasDistributedOperand(copyInCMXOp)) {
        return false;
    }

    const auto copyInInputType = mlir::cast<vpux::NDTypeInterface>(copyInCMXOp.getInput().getType());
    const auto copyInOutputType = mlir::cast<vpux::NDTypeInterface>(copyInCMXOp.getOutput().getType());
    if (copyInInputType.getMemoryKind() != VPU::MemoryKind::DDR ||
        copyInOutputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        return false;
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (copyBackToDDROp == nullptr || !copyBackToDDROp->hasOneUse() ||
        vpux::VPUIP::hasDistributedOperand(copyBackToDDROp)) {
        return false;
    }

    const auto copyBackInputType = mlir::cast<vpux::NDTypeInterface>(copyBackToDDROp.getInput().getType());
    const auto copyBackOutputType = mlir::cast<vpux::NDTypeInterface>(copyBackToDDROp.getOutput().getType());
    if (copyBackInputType.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
        copyBackOutputType.getMemoryKind() != VPU::MemoryKind::DDR) {
        return false;
    }

    auto copyToNCEOp = *(copyBackToDDROp->getUsers().begin());
    if (vpux::VPUIP::hasDistributedOperand(copyToNCEOp)) {
        const auto clusterOutput = VPUIP::getLayerOutputs(copyToNCEOp)[0];
        const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(clusterOutput.getType());

        VPUX_THROW_UNLESS(distributedType != nullptr, "Can not get distributed type of Cluster Copy Op");
        if (!isSplitContinuousBufferType(distributedType)) {
            return false;
        }

        const auto perAxisAttrs = VPUIP::getPerAxisTileSwKernelAttr(swKernelOp);
        VPUX_THROW_UNLESS(perAxisAttrs.has_value(), "Can not get PerAxisTile attribution");
        const auto repeateAxis = perAxisAttrs.value().axis;

        // If PerAxisTile Op repeate Axis same with Cluster Copy Tiling Axis
        // Should not fuse PerAxisTileDMA with Cluster Copy Op
        const auto distributionAttr = distributedType.getDistribution();
        if (distributionAttr.getNumTiles() != nullptr) {
            const auto isValidTile = [&](auto dim) {
                return dim > 1;
            };

            const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
            const auto iter = llvm::find_if(numTiles, isValidTile);
            if (iter != numTiles.end()) {
                const auto tilingAxis = std::distance(numTiles.begin(), iter);
                if (repeateAxis.getInt() == tilingAxis) {
                    return false;
                }
            }
        }
    }

    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(copyToNCEOp);
    if (childCopyOp == nullptr) {
        return false;
    }

    const auto childOutputType = mlir::cast<vpux::NDTypeInterface>(childCopyOp.getOutput().getType());
    return childOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
}

//
// FuseMemPermuteWithClusterCopy
//

// Copy(ddr->cmx)
//      |
// SW.kernel(memPermute)
//      |                                 ->        Clustering Tiling PermuteDMA (ddr->cmx)
// Copy (cmx->ddr)
//      |
// Clustering Ciling Copy (ddr->cmx)

class FuseMemPermuteWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseMemPermuteWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseMemPermuteWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseMemPermuteWithClusterCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (!checkPermuteWithCopyPattern(swKernelOp, _log)) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (copyBackToDDROp == nullptr || vpux::VPUIP::hasDistributedOperand(copyBackToDDROp)) {
        return mlir::failure();
    }

    auto user = *(copyBackToDDROp->getUsers().begin());
    if (!vpux::VPUIP::hasDistributedOperand(user)) {
        return mlir::failure();
    }

    _log.trace("Got Permute -> Cluster Copy pattern. MemPermute '{0}' at '{1}'", swKernelOp->getName(),
               swKernelOp->getLoc());

    // Check distribution mode
    const auto clusterOutput = VPUIP::getLayerOutputs(user)[0];
    const auto distributedType = mlir::cast<VPUIP::DistributedBufferType>(clusterOutput.getType());
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED &&
        !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).value();
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();

    if (vpux::VPUIP::hasDistributedOperand(copyInCMXOp)) {
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(user);
    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(user, copyInCMXOp.getInput(), VPUIP::getLayerOutputs(user)[0],
                                                     mlir::AffineMapAttr::get(memPerm), nullptr);

    _log.nest().trace("Wrap MemPermute '{0}' at '{1}' with next Cluster Copy.", swKernelOp->getName(),
                      swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(copyInCMXOp);

    return mlir::success();
}

//
// FuseMemPermuteWithCopy
//

// Copy(ddr->cmx)
//      |
// SW.kernel(memPermute)
//      |                      ->     VPUIP.PermuteDMA(ddr->cmx)
// Copy (cmx->ddr)
//      |
// Copy (ddr->cmx)

class FuseMemPermuteWithCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseMemPermuteWithCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseMemPermuteWithCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseMemPermuteWithCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (!checkPermuteWithCopyPattern(swKernelOp, _log)) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(copyBackToDDROp)) {
        return mlir::failure();
    }
    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(copyBackToDDROp->getUsers().begin()));
    if (childCopyOp == nullptr || vpux::VPUIP::hasDistributedOperand(childCopyOp)) {
        return mlir::failure();
    }

    _log.trace("Got Permute -> Copy pattern. MemPermute '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (vpux::VPUIP::hasDistributedOperand(copyInCMXOp)) {
        return mlir::failure();
    }
    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).value();
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(childCopyOp);
    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(childCopyOp, copyInCMXOp.getInput(), childCopyOp.getOutputBuff(),
                                                     mlir::AffineMapAttr::get(memPerm), nullptr);

    _log.nest().trace("Wrap MemPermute '{0}' at '{1}' with next copy.", swKernelOp->getName(), swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(copyInCMXOp);

    return mlir::success();
}

//
// FuseExpandWithClusterCopy
//

// Expand (U8)
//      |                                 ->      Cluster Tiling ExpandDMA (ddr->cmx)
// Cluster Tiling Copy (ddr->cmx)

class FuseExpandWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandWithClusterCopy::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (!checkExpandU8Pattern(expandOp, _log) && !checkExpandFP16Pattern(expandOp, _log)) {
        return mlir::failure();
    }

    auto user = *(expandOp->getUsers().begin());

    if (!vpux::VPUIP::hasDistributedOperand(user)) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> Cluster Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    // check distribution mode
    const auto clusterOutput = VPUIP::getLayerOutputs(user)[0];
    const auto distributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(clusterOutput.getType());
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED &&
        !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(user);
    rewriter.replaceOpWithNewOp<VPUIP::ExpandDMAOp>(user, expandOp.getInput(), VPUIP::getLayerOutputs(user)[0],
                                                    expandOp.getPadsBeginAttr(), expandOp.getPadsEndAttr(), nullptr);

    _log.nest().trace("Wrap Expand '{0}' at '{1}' with next Cluster Copy.", expandOp->getName(), expandOp->getLoc());

    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// FuseExpandWithCopy
//

// Expand (U8)
//      |                ->      ExpandDMA (ddr->cmx)
// Copy (ddr->cmx)

class FuseExpandWithCopy final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandWithCopy(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandWithCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandWithCopy::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (!checkExpandU8Pattern(expandOp, _log) && !checkExpandFP16Pattern(expandOp, _log)) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    auto expandCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(expandOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(expandCopyOutOp)) {
        return mlir::failure();
    }
    if (expandCopyOutOp == nullptr) {
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(expandCopyOutOp);
    rewriter.replaceOpWithNewOp<VPUIP::ExpandDMAOp>(expandCopyOutOp, expandOp.getInput(),
                                                    expandCopyOutOp.getOutputBuff(), expandOp.getPadsBeginAttr(),
                                                    expandOp.getPadsEndAttr(), nullptr);

    _log.nest().trace("Wrap Expand '{0}' at '{1}' with next Copy.", expandOp->getName(), expandOp->getLoc());

    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// FusePerAxisTileWithClusterCopy
//

// Copy(ddr->cmx)
//      |
// SW.kernel(PerAxisTile)
//      |                                 ->        Clustering Tiling PerAxisTileDMA (ddr->cmx)
// Copy (cmx->ddr)
//      |
// Clustering Ciling Copy (ddr->cmx)

class FusePerAxisTileWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FusePerAxisTileWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FusePerAxisTileWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FusePerAxisTileWithClusterCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    if (!checkPerAxisTilePattern(swKernelOp, _log)) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(copyBackToDDROp)) {
        return mlir::failure();
    }
    if (copyBackToDDROp == nullptr) {
        return mlir::failure();
    }

    auto user = *copyBackToDDROp->getUsers().begin();
    if (!vpux::VPUIP::hasDistributedOperand(user)) {
        return mlir::failure();
    }

    _log.trace("Got PerAxisTile -> Cluster Copy pattern. PerAxisTile '{0}' at '{1}'", swKernelOp->getName(),
               swKernelOp->getLoc());

    // Check distribution mode
    const auto clusterOutput = vpux::VPUIP::getLayerOutputs(user)[0];
    const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(clusterOutput.getType());
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED &&
        !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    auto perAxisTileAttrs = VPUIP::getPerAxisTileSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(perAxisTileAttrs.has_value(),
                      "Cannot extract PerAxisTile attribute from perAxisTile SwKernel '{0}'.", swKernelOp.getLoc());

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (vpux::VPUIP::hasDistributedOperand(copyInCMXOp)) {
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(user);
    rewriter.replaceOpWithNewOp<VPUIP::PerAxisTileDMAOp>(user, copyInCMXOp.getInput(), VPUIP::getLayerOutputs(user)[0],
                                                         perAxisTileAttrs.value().axis,
                                                         perAxisTileAttrs.value().repeats, nullptr);

    _log.nest().trace("Wrap PerAxisTile '{0}' at '{1}' with next Cluster Copy.", swKernelOp->getName(),
                      swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(copyInCMXOp);

    return mlir::success();
}

//
// FuseExpandWithUpsampling
//

class FuseExpandWithUpsampling final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandWithUpsampling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandWithUpsampling");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandWithUpsampling::matchAndRewrite(VPUIP::ExpandOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found ExpandOp Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto upsamplingOp = origOp.getInput().getDefiningOp<VPUIP::UpsamplingOp>();

    if (!upsamplingOp) {
        return mlir::failure();
    }

    if (!onlyExpandAtChannel(origOp)) {
        return mlir::failure();
    }

    _log.trace("Found ExpandOp Operation '{0}' at '{1}'", upsamplingOp->getName(), upsamplingOp->getLoc());
    const auto padChannel = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    auto padChannelAttr = getIntArrayAttr(upsamplingOp.getContext(), padChannel);

    const auto outputShape = getShape(origOp.getOutput());

    const auto upsamplingFactorVectorTmp = parseIntArrayAttr<int64_t>(upsamplingOp.getUpsamplingFactor());
    SmallVector<int64_t> upsamplingFactorVector = {1, upsamplingFactorVectorTmp[2], upsamplingFactorVectorTmp[1],
                                                   upsamplingFactorVectorTmp[0]};
    const auto inputType = mlir::cast<NDTypeInterface>(upsamplingOp.getInput().getType());
    const auto zeroType =
            mlir::cast<NDTypeInterface>(mlir::MemRefType::get(outputShape.raw(), inputType.getElementType()))
                    .changeDimsOrder(inputType.getDimsOrder());
    auto constZeros = Const::createZerosConst(rewriter, origOp.getLoc(), mlir::cast<mlir::MemRefType>(zeroType));

    auto copyZeroOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), constZeros, origOp.getOutputBuff());
    if (vpux::VPUIP::hasDistributedOperand(copyZeroOp)) {
        return mlir::failure();
    }
    auto upsampleFactorAttr = getIntArrayAttr(origOp.getContext(), upsamplingFactorVector);

    auto upsampleDMA = rewriter.replaceOpWithNewOp<VPUIP::UpsamplingDMAOp>(
            origOp, upsamplingOp.getInput(), copyZeroOp.getOutput(), upsampleFactorAttr, /*dma_descriptor,*/ nullptr,
            padChannelAttr, getIntAttr(origOp->getContext(), 0), /*is_out_of_order*/ nullptr,
            /*is_critical*/ nullptr, /*dmaHwpId=*/nullptr,
            /*profilingMetadata=*/nullptr);

    rewriter.eraseOp(upsamplingOp);

    _log.trace("Create new upsampling operation '{0}'", upsampleDMA);
    return mlir::success();
}

//
// FusePerAxisTileWithCopy
//

// Copy(ddr->cmx)
//      |
// SW.kernel(PerAxisTile)
//      |                      ->     VPUIP.PerAxisTileDMA(ddr->cmx)
// Copy (cmx->ddr)
//      |
// Copy (ddr->cmx)

class FusePerAxisTileWithCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FusePerAxisTileWithCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FusePerAxisTileWithCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FusePerAxisTileWithCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (!checkPerAxisTilePattern(swKernelOp, _log)) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(copyBackToDDROp)) {
        return mlir::failure();
    }
    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(copyBackToDDROp->getUsers().begin()));
    if (childCopyOp == nullptr || vpux::VPUIP::hasDistributedOperand(childCopyOp)) {
        return mlir::failure();
    }

    _log.trace("Got PerAxisTile -> Copy pattern. PerAxisTile '{0}' at '{1}'", swKernelOp->getName(),
               swKernelOp->getLoc());

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (vpux::VPUIP::hasDistributedOperand(copyInCMXOp)) {
        return mlir::failure();
    }
    auto perAxisTileAttrs = VPUIP::getPerAxisTileSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(perAxisTileAttrs.has_value(),
                      "Cannot extract perAxisTile attribute from perAxisTile SwKernel '{0}'.", swKernelOp.getLoc());

    rewriter.setInsertionPointAfter(childCopyOp);
    rewriter.replaceOpWithNewOp<VPUIP::PerAxisTileDMAOp>(childCopyOp, copyInCMXOp.getInput(),
                                                         childCopyOp.getOutputBuff(), perAxisTileAttrs.value().axis,
                                                         perAxisTileAttrs.value().repeats, nullptr);

    _log.nest().trace("Wrap PerAxisTile '{0}' at '{1}' with next copy.", swKernelOp->getName(), swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(copyInCMXOp);

    return mlir::success();
}

//
// FuseExpandAndPermuteWithClusterCopy
//

// Expand (U8)
//      |
// Copy(ddr->cmx)
//      |
// SW.kernel(memPermute)          ->       Clustering Tiling PermuteDMA (ddr->cmx)
//      |
// Copy (cmx->ddr)
//      |
// Clustering Tiling Copy (ddr->cmx)

class FuseExpandAndPermuteWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandAndPermuteWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandAndPermuteWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandAndPermuteWithClusterCopy::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    if (!checkExpandWithPermutePattern(expandOp, _log)) {
        return mlir::failure();
    }

    auto expandCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(expandOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(expandCopyOutOp)) {
        return mlir::failure();
    }
    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(expandCopyOutOp->getUsers().begin()));

    auto permuteCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(permuteCopyOutOp)) {
        return mlir::failure();
    }
    auto user = *(permuteCopyOutOp->getUsers().begin());
    if (user == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> permute -> Cluster Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(),
               expandOp->getLoc());

    // check distribution mode
    const auto clusterCopyOutput = VPUIP::getLayerOutputs(user)[0];
    const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(clusterCopyOutput.getType());
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED &&
        !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).value();
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(user);
    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(user, expandOp.getInput(), VPUIP::getLayerOutputs(user)[0],
                                                     mlir::AffineMapAttr::get(memPerm), nullptr);

    _log.nest().trace("Wrap Expand '{0}' at '{1}' and MemPermute with next Cluster Copy.", expandOp->getName(),
                      expandOp->getLoc());

    rewriter.eraseOp(permuteCopyOutOp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(expandCopyOutOp);
    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// FuseExpandAndPermuteWithCopy
//

// Expand (U8)
//      |
// Copy(ddr->cmx)
//      |
// SW.kernel(memPermute)         ->      PermuteDMA (ddr->cmx)
//      |
// Copy (cmx->ddr)
//      |
// Copy (ddr->cmx)

class FuseExpandAndPermuteWithCopy final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandAndPermuteWithCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandAndPermuteWithCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandAndPermuteWithCopy::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!checkExpandWithPermutePattern(expandOp, _log)) {
        return mlir::failure();
    }

    auto expandCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(expandOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(expandCopyOutOp)) {
        return mlir::failure();
    }
    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(expandCopyOutOp->getUsers().begin()));

    auto permuteCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(permuteCopyOutOp)) {
        return mlir::failure();
    }
    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(permuteCopyOutOp->getUsers().begin()));
    if (childCopyOp == nullptr || vpux::VPUIP::hasDistributedOperand(childCopyOp)) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> Permute -> Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    rewriter.setInsertionPointAfter(childCopyOp);
    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).value();
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(childCopyOp, expandOp.getInput(), childCopyOp.getOutputBuff(),
                                                     mlir::AffineMapAttr::get(memPerm), nullptr);

    _log.nest().trace("Wrap Expand '{0}' at '{1}' and MemPermute with next Copy.", expandOp->getName(),
                      expandOp->getLoc());

    rewriter.eraseOp(permuteCopyOutOp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(expandCopyOutOp);
    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// FuseSpaceToDepthAndPermute
//

// SW.Kernel(SpaceToDepth, Layout0->Layout1)
//      |
// Copy(cmx->ddr)
//      |                                             ->      SpaceToDepthDMA(Layout0->Layout2)
// Copy(ddr->cmx)
//      |
// SW.Kernel(MemPermute(Reorder), Layout1->Layout2)

class FuseSpaceToDepthAndPermute final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseSpaceToDepthAndPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseSpaceToDepthAndPermute");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseSpaceToDepthAndPermute::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got SWKernel '{0}' at '{1}'.", swKernelOp->getName(), swKernelOp->getLoc());
    if (!checkSpaceToDepthWithPermutePattern(swKernelOp, _log.nest())) {
        _log.nest().trace("Fuse SpaceToDepth and Permute pattern mismatch.");
        return mlir::failure();
    }

    auto s2dCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(s2dCopyOutOp)) {
        return mlir::failure();
    }
    auto permuteCopyInOp = mlir::dyn_cast<VPUIP::CopyOp>(*(s2dCopyOutOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(permuteCopyInOp)) {
        return mlir::failure();
    }
    auto permuteSwKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(permuteCopyInOp->getUsers().begin()));

    const auto s2dInType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getInputs().front().getType());
    const auto s2dOutType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getOutputs().front().getType());
    const auto permuteInType = mlir::cast<vpux::NDTypeInterface>(permuteSwKernelOp.getInputs().front().getType());
    const auto permuteOutType = mlir::cast<vpux::NDTypeInterface>(permuteSwKernelOp.getOutputs().front().getType());

    const auto inOrder = s2dInType.getDimsOrder();
    const auto outOrder = permuteOutType.getDimsOrder();

    if (!(inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NHWC)) {
        _log.nest().trace("SpaceToDepthDMA do not support layout '{0}'->'{1}'", inOrder, outOrder);
        return mlir::failure();
    }

    auto spaceToDepthAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(spaceToDepthAttrs.has_value(),
                      "Cannot extract SpaceToDepth attributes from SpaceToDepth SwKernel '{0}'.", swKernelOp.getLoc());
    auto modeAttr = spaceToDepthAttrs.value().first;
    auto blockSizeAttr = spaceToDepthAttrs.value().second;

    _log.nest().trace("Wrap SpaceToDepth('{0}'->'{1}') and MemPermute('{2}'->'{3}') as SpaceToDepthDMA('{4}'->'{5}')",
                      s2dInType.getDimsOrder(), s2dOutType.getDimsOrder(), permuteInType.getDimsOrder(),
                      permuteOutType.getDimsOrder(), inOrder, outOrder);

    auto input = swKernelOp.getOperand(0);

    auto outputMemRef = mlir::cast<mlir::MemRefType>(permuteOutType);
    auto allocSpaceToDepthOp = rewriter.create<mlir::memref::AllocOp>(permuteSwKernelOp->getLoc(), outputMemRef);

    rewriter.replaceOpWithNewOp<VPUIP::SpaceToDepthDMAOp>(permuteSwKernelOp, input, allocSpaceToDepthOp, blockSizeAttr,
                                                          modeAttr, nullptr);

    rewriter.eraseOp(permuteCopyInOp);
    rewriter.eraseOp(s2dCopyOutOp);
    rewriter.eraseOp(swKernelOp);

    return mlir::success();
}

//
// WrapDepthToSpaceAsClusterNNDMA
//

// Match this pattern to convert SWKernel DepthToSpace to
// multi-cluster DepthToSpaceDMA.
//
//   --- (Optional if no cluster output) ---
//   |        ClusterCopy(cmx->ddr)        |
//   |                 |                   |
//   |           Copy(ddr->cmx)            |
//   -----------       |         -----------
//            SWKernel(DepthToSpace)
//                     |
//   ----------- Copy(cmx->ddr)  -----------
//   |                 |                   |
//   |            [ShapeCast]              |
//   |                 |                   |
//   |        ClusterCopy(ddr->cmx)        |
//   --- (Optional if no cluster input)  ---

class WrapDepthToSpaceAsClusterNNDMA final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    WrapDepthToSpaceAsClusterNNDMA(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("WrapDepthToSpaceAsClusterNNDMA");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    enum PatternType { NONE, INPUT, OUTPUT, BOTH };
    PatternType checkPattern(VPUIP::SwKernelOp swKernelOp) const;
    bool isValidConversion(VPUIP::SwKernelOp swKernelOp) const;

private:
    Logger _log;
};

bool hasValidInputPerClusterShape(VPUIP::SwKernelOp swKernelOp, VPUIP::DistributedBufferType dmaDistributedType) {
    // Extract D2S attributes
    auto d2sAttrs = VPUIP::getDepthToSpaceSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(d2sAttrs.has_value(), "Failed to extract attributes from DepthToSpace SwKernel '{0}'.",
                      swKernelOp.getLoc());

    auto blockSize = std::get<1>(d2sAttrs.value()).getInt();
    auto paddedChannels = std::get<2>(d2sAttrs.value());
    auto paddedIC = paddedChannels ? paddedChannels.getInput().getInt() : 0;
    auto paddedOC = paddedChannels ? paddedChannels.getOutput().getInt() : 0;

    const auto inputShapes =
            SmallVector<Shape>(llvm::map_range(dmaDistributedType.getPerClusterMemoryShapes(), [&](ShapeRef outShape) {
                return VPUIP::backInferD2SInputShape(outShape.toValues(), paddedOC, paddedIC, blockSize);
            }));

    auto hasZeroDim = [](const Shape shape) {
        return std::any_of(shape.begin(), shape.end(), [](int64_t value) {
            return value == 0;
        });
    };

    return std::none_of(inputShapes.begin(), inputShapes.end(), hasZeroDim);
}

bool WrapDepthToSpaceAsClusterNNDMA::isValidConversion(VPUIP::SwKernelOp swKernelOp) const {
    _log.trace("Checking DepthToSpaceAsMultiCluster pattern.");

    if (!VPUIP::isDepthToSpaceSwKernel(swKernelOp)) {
        _log.nest().trace("SWKernel is not DepthToSpace.");
        return false;
    }

    _log.nest().trace("Got DepthToSpace SwKernel '{0}' at '{1}'.", swKernelOp->getName(), swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, _log.nest())) {
        _log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

    const auto d2sInType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getInputs().front().getType());
    const auto d2sOutType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getOutputs().front().getType());
    const auto inOrder = d2sInType.getDimsOrder();
    const auto outOrder = d2sOutType.getDimsOrder();
    if (inOrder != DimsOrder::NHWC || outOrder != DimsOrder::NHWC) {
        _log.nest().trace("Only support NHWC->NHWC, but got: '{0}'->'{1}'", inOrder, outOrder);
        return false;
    }

    auto d2sAttrs = VPUIP::getDepthToSpaceSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(d2sAttrs.has_value(), "Cannot extract DepthToSpace attributes from SwKernel '{0}'.",
                      swKernelOp.getLoc());
    auto mode = std::get<0>(d2sAttrs.value()).getValue();
    if (mode != IE::DepthToSpaceMode::BLOCKS_FIRST) {
        _log.nest().trace("Only support BlocksFirst mode");
        return false;
    }
    return true;
}

WrapDepthToSpaceAsClusterNNDMA::PatternType WrapDepthToSpaceAsClusterNNDMA::checkPattern(
        VPUIP::SwKernelOp swKernelOp) const {
    if (!isValidConversion(swKernelOp)) {
        return PatternType::NONE;
    }

    const auto isSegmented = [&](vpux::NDTypeInterface operandType) {
        auto operandDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
        VPUX_THROW_WHEN(operandDistType == nullptr, "Operand is not distributed type");
        const auto distribution = operandDistType.getDistribution();
        const auto mode = distribution.getMode().getValue();
        return mode == VPU::DistributionMode::SEGMENTED;
    };

    const auto isValidDistributedCopyOp = [&](mlir::Operation* op, bool isCopyIn) {
        if (!vpux::VPUIP::hasDistributedOperand(op)) {
            return false;
        }
        auto const input = VPUIP::getLayerInputs(op)[0];
        auto const output = VPUIP::getLayerOutputs(op)[0];
        const auto childCopyType = isCopyIn ? mlir::dyn_cast<vpux::NDTypeInterface>(input.getType())
                                            : mlir::dyn_cast<vpux::NDTypeInterface>(output.getType());
        if (childCopyType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
            return false;
        }

        const auto clusterOperand = isCopyIn ? input : output;
        const auto clusterOperandType = mlir::cast<vpux::NDTypeInterface>(clusterOperand.getType());
        return isSegmented(clusterOperandType) &&
               isSplitContinuousBufferType(mlir::dyn_cast<VPUIP::DistributedBufferType>(clusterOperandType));
    };

    const auto isValidCopyOp = [&](VPUIP::CopyOp copyOp, bool isCopyIn) {
        const auto copyInputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getInput().getType());
        const auto copyOutputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutput().getType());
        return isCopyIn ? (copyInputType.getMemoryKind() == VPU::MemoryKind::DDR &&
                           copyOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN)
                        : (copyInputType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
                           copyOutputType.getMemoryKind() == VPU::MemoryKind::DDR);
    };

    const auto findInputClusterCopy = [&](VPUIP::SwKernelOp swKernelOp) -> mlir::Operation* {
        auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
        if (vpux::VPUIP::hasDistributedOperand(copyInCMXOp)) {
            return nullptr;
        }

        if (copyInCMXOp == nullptr || !copyInCMXOp->hasOneUse() || !isValidCopyOp(copyInCMXOp, true)) {
            return nullptr;
        }
        const auto copyInInputType = mlir::cast<vpux::NDTypeInterface>(copyInCMXOp.getInput().getType());
        const auto copyInOutputType = mlir::cast<vpux::NDTypeInterface>(copyInCMXOp.getOutput().getType());
        if (copyInInputType.getMemoryKind() != VPU::MemoryKind::DDR ||
            copyInOutputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
            return nullptr;
        }
        auto parent = copyInCMXOp.getOperand(0).getDefiningOp();
        if (parent == nullptr || !parent->hasOneUse() || !isValidDistributedCopyOp(parent, true)) {
            return nullptr;
        }
        return parent;
    };

    const auto findOutputClusterCopy = [&](VPUIP::SwKernelOp swKernelOp) -> mlir::Operation* {
        if (!swKernelOp->hasOneUse()) {
            return nullptr;
        }
        auto copyOutDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
        if (copyOutDDROp == nullptr || !copyOutDDROp->hasOneUse() || vpux::VPUIP::hasDistributedOperand(copyOutDDROp) ||
            !isValidCopyOp(copyOutDDROp, false)) {
            return nullptr;
        }
        auto user = *copyOutDDROp->getUsers().begin();
        while (VPUIP::isPureViewOp(user)) {
            if (!user->hasOneUse()) {
                return nullptr;
            }
            user = *user->getUsers().begin();
        }
        if (!isValidDistributedCopyOp(user, false)) {
            return nullptr;
        }
        return user;
    };

    const auto clusterCopyIn = findInputClusterCopy(swKernelOp);
    const auto clusterCopyOut = findOutputClusterCopy(swKernelOp);

    if (clusterCopyIn == nullptr && clusterCopyOut == nullptr) {
        _log.nest().trace("Neither input nor output is in multicluster");
        return PatternType::NONE;
    } else if (clusterCopyIn != nullptr && clusterCopyOut == nullptr) {
        _log.nest().trace("Found input in multicluster");
        return PatternType::INPUT;
    } else if (clusterCopyIn == nullptr && clusterCopyOut != nullptr) {
        _log.nest().trace("Found output in multicluster");
        return PatternType::OUTPUT;
    } else {
        _log.nest().trace("Found both input and output in multicluster");
        return PatternType::BOTH;
    }
}

mlir::LogicalResult WrapDepthToSpaceAsClusterNNDMA::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    const auto patternType = checkPattern(swKernelOp);
    if (patternType == PatternType::NONE) {
        return mlir::failure();
    }

    _log.trace("Found DepthToSpace at '{0}' with ClusterCopy pattern", swKernelOp->getLoc());

    auto ctx = swKernelOp.getContext();
    auto arch = VPU::getArch(swKernelOp.getOperation());

    // Extract D2S attributes
    auto d2sAttrs = VPUIP::getDepthToSpaceSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(d2sAttrs.has_value(), "Failed to extract attributes from DepthToSpace SwKernel '{0}'.",
                      swKernelOp.getLoc());
    auto modeAttr = std::get<0>(d2sAttrs.value());
    auto blockSizeAttr = std::get<1>(d2sAttrs.value());
    auto paddedChannels = std::get<2>(d2sAttrs.value());

    // Extract D2S output
    auto d2sOutputBuff = swKernelOp.getOperand(1);
    auto d2sOutputBuffType = mlir::cast<vpux::NDTypeInterface>(d2sOutputBuff.getType());

    const auto tileCount = getIntAttr(ctx, VPUIP::getNumTilesUsed(swKernelOp->getParentOfType<mlir::ModuleOp>()));

    const auto insertionPoint = patternType == PatternType::OUTPUT || patternType == PatternType::BOTH
                                        ? *swKernelOp->getUsers().begin()
                                        : swKernelOp.getOperation();
    rewriter.setInsertionPointAfter(insertionPoint);

    SmallVector<mlir::Operation*> opsToErase;

    // If pattern is INPUT or BOTH, which means the input side is in multicluster,
    // we need to create an input cluster copy before ClusterD2SDMAOp
    auto d2sInput = swKernelOp.getOperand(0);
    if (patternType == PatternType::INPUT || patternType == PatternType::BOTH) {
        auto inputCopyOp = d2sInput.getDefiningOp<VPUIP::CopyOp>();
        if (vpux::VPUIP::hasDistributedOperand(inputCopyOp)) {
            return mlir::failure();
        }
        VPUX_THROW_WHEN(inputCopyOp == nullptr, "Failed to get input copy of DepthToSpace");
        auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(swKernelOp);
        auto clusterCopyInAllocType = createDMADistributedTensorType(ctx, inputCopyOp.getOutput().getType(), tileCount,
                                                                     arch, uniformDistributedSegments);
        auto clusterCopyInAllocOp = rewriter.create<VPURT::AllocDistributed>(inputCopyOp.getLoc(),
                                                                             clusterCopyInAllocType, nullptr, nullptr);
        opsToErase.push_back(inputCopyOp);
        auto d2sCopyInOp = rewriter.create<VPUIP::CopyOp>(inputCopyOp.getLoc(), clusterCopyInAllocType,
                                                          inputCopyOp.getInput(), clusterCopyInAllocOp.getBuffer());
        d2sInput = d2sCopyInOp.getResult();
        _log.nest().trace("Create new cluster copy-in op: {0}", d2sCopyInOp);
    }

    // If pattern is OUTPUT or BOTH, which means the output side is in multicluster,
    // we need to create ClusterD2SDMAOp with a following output cluster copy
    // Otherwise, when the pattern is INPUT, which means the output side is not in multicluster,
    // we only need to create a ClusterD2SDMAOp
    // The created ClusterD2SDMAOp will be unrolled according to its input/output type later in
    // UnrollDepthToSpaceDMAPass
    if (patternType == PatternType::OUTPUT || patternType == PatternType::BOTH) {
        auto outputCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*swKernelOp->getUsers().begin());
        if (vpux::VPUIP::hasDistributedOperand(outputCopyOp)) {
            return mlir::failure();
        }
        VPUX_THROW_WHEN(outputCopyOp == nullptr, "Failed to get output copy of DepthToSpace");
        auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(swKernelOp);
        // create d2s
        auto d2sOutAllocType =
                createDMADistributedTensorType(ctx, d2sOutputBuffType, tileCount, arch, uniformDistributedSegments);
        // E#82228 When unrolling we should have valid per cluster input buffers, these checks prevent wrapping when the
        // input will no longer be valid Example, with 1x1x9x9 the per cluster shape with alignment would be 1x1x8x9 and
        // 1x1x1x9 for 1x1x1x9 the input buffer would be 1x9x0x3 which is invalid, hence such cases should run on shave
        if (!hasValidInputPerClusterShape(swKernelOp, d2sOutAllocType)) {
            return mlir::failure();
        }
        auto d2sOutAllocOp =
                rewriter.create<VPURT::AllocDistributed>(swKernelOp.getLoc(), d2sOutAllocType, nullptr, nullptr);
        auto d2sOp = rewriter.create<VPUIP::DepthToSpaceDMAOp>(swKernelOp.getLoc(), d2sInput, d2sOutAllocOp.getBuffer(),
                                                               blockSizeAttr, modeAttr, nullptr, paddedChannels);
        _log.nest().trace("Create new cluster DepthToSpaceDMAOp: {0}", d2sOp);
        opsToErase.push_back(swKernelOp);
        // create output copy
        auto d2sCopyOutOp = rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(
                outputCopyOp, outputCopyOp.getOutputBuff().getType(), d2sOp.getResult(), outputCopyOp.getOutputBuff());
        _log.nest().trace("Create new cluster copy-out op: {0}", d2sCopyOutOp);
    } else {
        // create d2s
        auto d2sOp = rewriter.replaceOpWithNewOp<VPUIP::DepthToSpaceDMAOp>(
                swKernelOp, d2sInput, d2sOutputBuff, blockSizeAttr, modeAttr, nullptr, paddedChannels);
        _log.nest().trace("Create new cluster DepthToSpaceDMAOp: {0}", d2sOp);
    }

    while (!opsToErase.empty()) {
        _log.nest().trace("Erase Op: {0}", opsToErase.back()->getLoc());
        rewriter.eraseOp(opsToErase.back());
        opsToErase.pop_back();
    }

    return mlir::success();
}

//
// FuseSpaceToDepthWithClusterCopy
//

// SW.kernel(spaceToDepth)
//      |                                 ->        Clustering Tiling SpaceToDepthDMA (cmx->cmx)
// Copy (cmx->ddr)
//      |
// Clustering Ciling Copy (ddr->cmx)

class FuseSpaceToDepthWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseSpaceToDepthWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseSpaceToDepthWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseSpaceToDepthWithClusterCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got SWKernel '{0}' at '{1}'.", swKernelOp->getName(), swKernelOp->getLoc());
    if (!checkSpaceToDepthPattern(swKernelOp, _log.nest())) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (vpux::VPUIP::hasDistributedOperand(copyBackToDDROp)) {
        return mlir::failure();
    }
    auto user = *(copyBackToDDROp->getUsers().begin());

    _log.nest().trace("Got SpaceToDepth -> Cluster Copy pattern. SpaceToDepth '{0}' at '{1}'", swKernelOp->getName(),
                      swKernelOp->getLoc());

    // Check distribution mode
    const auto clusterOutput = VPUIP::getLayerOutputs(user)[0];
    const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(clusterOutput.getType());
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    auto s2dAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
    auto modeAttr = s2dAttrs.value().first;
    auto blockSizeAttr = s2dAttrs.value().second;

    const auto distributionAttr = distributedType.getDistribution();

    // Currently only support SEGMENTED or OVERLAPPED over H
    if (!isSegmentedOverH(distributionAttr) && !isOverlappedOverH(distributionAttr)) {
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(user);
    rewriter.replaceOpWithNewOp<VPUIP::SpaceToDepthDMAOp>(
            user, swKernelOp.getOperand(0), VPUIP::getLayerOutputs(user)[0], blockSizeAttr, modeAttr, nullptr);

    _log.nest().trace("Wrap SpaceToDepth '{0}' at '{1}' with next Cluster Copy.", swKernelOp->getName(),
                      swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);

    return mlir::success();
}

//
// FuseClusterCopyWithMemPermute
//

// Duplicated Tiling Copy(cmx->ddr)
//              |
//         Copy(ddr->cmx)                ->        Clustering Tiling PermuteDMA (cmx->cmx)
//              |
//         SW.kernel(memPermute)
//              |

class FuseClusterCopyWithMemPermute final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseClusterCopyWithMemPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseClusterCopyWithMemPermute");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseClusterCopyWithMemPermute::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (VPUIP::hasBoundedBuffers(swKernelOp) || VPUIP::hasUngroupedBoundedBuffers(swKernelOp)) {
        return mlir::failure();
    }

    if (!checkTilingCopyWithPermutePattern(swKernelOp, _log)) {
        return mlir::failure();
    }
    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (vpux::VPUIP::hasDistributedOperand(copyInCMXOp)) {
        return mlir::failure();
    }
    auto parent = copyInCMXOp->getOperand(0).getDefiningOp();
    VPUX_THROW_WHEN(copyInCMXOp == nullptr || parent == nullptr || !vpux::VPUIP::hasDistributedOperand(parent),
                    "Invalid copy");

    _log.trace("Process sw kernel op {0}", swKernelOp);

    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).value();
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(swKernelOp, VPUIP::getLayerInputs(parent)[0],
                                                     VPUIP::getLayerOutputs(swKernelOp)[0],
                                                     mlir::AffineMapAttr::get(memPerm), nullptr);
    rewriter.eraseOp(copyInCMXOp);
    rewriter.eraseOp(parent);

    return mlir::success();
}

//
// FuseClusterMemPermuteWithViewLikeOps
//

//  Cluster Tiling PermuteDMA (cmx->cmx)
//              |
//          ViewLikeOp                          Clustering Tiling PermuteDMA (cmx->cmx)
//              |                       ->                 |
//         Copy (cmx->ddr)                              ViewLikeOp
//              |
// Duplicated Tiling Copy (ddr->cmx)

class FuseClusterMemPermuteWithViewLikeOps final : public mlir::OpRewritePattern<VPUIP::PermuteDMAOp> {
public:
    FuseClusterMemPermuteWithViewLikeOps(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::PermuteDMAOp>(ctx), _log(log) {
        setDebugName("FuseClusterMemPermuteWithViewLikeOps");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::DistributionInfoAttr getDuplicatedDistribution(ShapeRef shape, VPU::DistributionInfoAttr origDistribution,
                                                        mlir::MLIRContext* ctx) const;

private:
    Logger _log;
};

VPU::DistributionInfoAttr FuseClusterMemPermuteWithViewLikeOps::getDuplicatedDistribution(
        ShapeRef shape, VPU::DistributionInfoAttr origDistribution, mlir::MLIRContext* ctx) const {
    const auto distrModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
    if (!isDistributedAttrWithExplicitShapesAndOffsets(origDistribution)) {
        return VPU::DistributionInfoAttr::get(
                ctx, distrModeAttr, nullptr, nullptr, nullptr, nullptr, origDistribution.getNumClusters(), nullptr,
                origDistribution.getUniformDistributedSegments(), nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    return VPU::getNonOverlappedDistributedAttr(shape, distrModeAttr, nullptr, origDistribution.getNumClusters(),
                                                nullptr, origDistribution.getUniformDistributedSegments(), ctx);
}

mlir::LogicalResult FuseClusterMemPermuteWithViewLikeOps::matchAndRewrite(VPUIP::PermuteDMAOp permuteOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("FuseClusterMemPermuteWithViewLikeOps: Found PermuteDMAOp {0}", permuteOp);
    if (!permuteOp->hasOneUse()) {
        return mlir::failure();
    }
    auto viewLikeOps = getPureViewLikeOpChains(permuteOp);

    // check copy op after viewLikeOps
    auto userOp = viewLikeOps.empty() ? *permuteOp->getUsers().begin() : *viewLikeOps.back()->getUsers().begin();
    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(userOp);
    if (copyOp == nullptr || !copyOp->hasOneUse() || vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return mlir::failure();
    }
    const auto copyInType = mlir::cast<vpux::NDTypeInterface>(copyOp.getInput().getType());
    const auto copyOutType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutput().getType());
    if (copyInType.getMemoryKind() != VPU::MemoryKind::CMX_NN || copyOutType.getMemoryKind() != VPU::MemoryKind::DDR) {
        return mlir::failure();
    }

    // check tiling copy op
    auto user = mlir::dyn_cast<VPUIP::CopyOp>(*copyOp->getUsers().begin());
    if (user == nullptr || !vpux::VPUIP::hasDistributedOperand(user)) {
        return mlir::failure();
    }
    const auto tilingCopyOutType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(user->getResult(0).getType());
    auto outDistribution = tilingCopyOutType.getDistribution();
    auto mode = outDistribution.getMode().getValue();
    if (tilingCopyOutType.getMemoryKind() != VPU::MemoryKind::CMX_NN || mode != VPU::DistributionMode::DUPLICATED) {
        return mlir::failure();
    }

    const auto outReqs = StrideReqs::compact(tilingCopyOutType.getRank());
    if (!outReqs.checkStrides(tilingCopyOutType)) {
        _log.trace("Skip complex case: output is strided");
        return mlir::failure();
    }

    // create new Cluster Tiling PermuteDMA with distributed output
    rewriter.setInsertionPointAfter(user);
    auto memPerm = permuteOp.getMemPermAttr();
    const auto ctx = permuteOp->getContext();
    const auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(permuteOp.getOutput().getType());
    const auto outShape = outType.getShape();
    const auto outElemType = outType.getElementType();
    const auto order = mlir::AffineMapAttr::get(outType.getDimsOrder().toAffineMap(ctx));
    auto permuteDistribution = getDuplicatedDistribution(outShape, outDistribution, ctx);
    auto newPermuteDistributedOutType = VPUIP::DistributedBufferType::get(
            ctx, outShape.raw(), outElemType, order, tilingCopyOutType.getMemSpace(), permuteDistribution);

    auto newAlloc = rewriter.create<VPURT::AllocDistributed>(permuteOp->getLoc(), newPermuteDistributedOutType, nullptr,
                                                             nullptr);

    auto newPermuteOp = rewriter.create<VPUIP::PermuteDMAOp>(permuteOp->getLoc(), VPUIP::getLayerInputs(permuteOp)[0],
                                                             newAlloc, memPerm, nullptr);
    _log.trace("create new cluster tiling permute op {0}", newPermuteOp);

    // create new view like ops
    auto newOutput = newPermuteOp->getResult(0);
    for (auto viewLikeOp : viewLikeOps) {
        mlir::IRMapping mapper;
        mapper.map(viewLikeOp->getOperands(), ArrayRef({newOutput}));
        auto* newViewLikeOp = rewriter.clone(*viewLikeOp, mapper);

        auto viewLikeOutType = mlir::cast<vpux::NDTypeInterface>(viewLikeOp->getResult(0).getType());
        auto viewLikeOutShape = viewLikeOutType.getShape();
        auto viewLikeOutOrder = mlir::AffineMapAttr::get(viewLikeOutType.getDimsOrder().toAffineMap(ctx));
        auto viewLikeElemType = viewLikeOutType.getElementType();
        auto viewLikeDistribution = getDuplicatedDistribution(viewLikeOutShape, outDistribution, ctx);
        auto newViewLikeOutType =
                VPUIP::DistributedBufferType::get(ctx, viewLikeOutShape.raw(), viewLikeElemType, viewLikeOutOrder,
                                                  tilingCopyOutType.getMemSpace(), viewLikeDistribution);
        newViewLikeOp->getResult(0).setType(newViewLikeOutType);
        newOutput = newViewLikeOp->getResult(0);

        viewLikeOp->dropAllUses();
        rewriter.eraseOp(viewLikeOp);
    }

    // There are instances where the tilingCopyOutType can have the alignment attribute set.
    // However, given that this pass currently applies just for DUPLICATED mode, it is safe to discard
    // the attribute. Alternatively, it could be tried to propagate the alignment up, but propagation
    // through GenericReshape/ShapeCast ops is not trivial.
    auto newOutputDistributedType = mlir::cast<vpux::VPUIP::DistributedBufferType>(newOutput.getType());
    if (tilingCopyOutType != newOutputDistributedType) {
        if (VPU::isDistributedCastCompatible(newOutputDistributedType, tilingCopyOutType).failed()) {
            _log.trace("View op's output type incompatible with consumer NCE.ClusterTiling input type: output = {0}, "
                       "input = {1}",
                       newOutputDistributedType, tilingCopyOutType);
            return mlir::failure();
        }

        auto distributedCastOp =
                rewriter.create<VPUIP::DistributedCastOp>(permuteOp->getLoc(), tilingCopyOutType, newOutput);
        newOutput = distributedCastOp->getResult(0);
    }

    rewriter.replaceOp(user, newOutput);
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(permuteOp);
    return mlir::success();
}

//
// WrapWithPermuteAsNNDMAPass
//

class WrapWithPermuteAsNNDMAPass final : public VPUIP::impl::WrapWithPermuteAsNNDMABase<WrapWithPermuteAsNNDMAPass> {
public:
    explicit WrapWithPermuteAsNNDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

// TODO: #71565
void WrapWithPermuteAsNNDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseExpandAndPermuteWithClusterCopy>(&ctx, _log);
    patterns.add<FuseExpandAndPermuteWithCopy>(&ctx, _log);
    patterns.add<FuseExpandWithClusterCopy>(&ctx, _log);
    patterns.add<FuseExpandWithCopy>(&ctx, _log);
    patterns.add<FuseExpandWithUpsampling>(&ctx, _log);
    patterns.add<FuseMemPermuteWithClusterCopy>(&ctx, _log);
    patterns.add<FuseMemPermuteWithCopy>(&ctx, _log);
    patterns.add<FuseClusterCopyWithMemPermute>(&ctx, _log);
    patterns.add<FuseClusterMemPermuteWithViewLikeOps>(&ctx, _log);
    patterns.add<FusePerAxisTileWithClusterCopy>(&ctx, _log);
    patterns.add<FusePerAxisTileWithCopy>(&ctx, _log);
    patterns.add<FuseSpaceToDepthAndPermute>(&ctx, _log);
    patterns.add<FuseSpaceToDepthWithClusterCopy>(&ctx, _log);
    patterns.add<WrapDepthToSpaceAsClusterNNDMA>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createWrapWithPermuteAsNNDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createWrapWithPermuteAsNNDMAPass(Logger log) {
    return std::make_unique<WrapWithPermuteAsNNDMAPass>(log);
}
