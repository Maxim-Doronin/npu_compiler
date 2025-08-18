//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux::arch37xx {

NDTypeInterface changeShape(NDTypeInterface originType, ShapeRef shape, ShapeRef offset);
NDTypeInterface getPerClusterInputType(NDTypeInterface inputType, NDTypeInterface outputType, mlir::AffineMap memPerm,
                                       ShapeRef outShape, ShapeRef offset);

class UnrollMultiClusterPermuteDMA {
public:
    static mlir::LogicalResult unrollSegmentedOrOverlappedOutput(VPUIP::PermuteDMAOp permuteOp,
                                                                 VPUIP::DistributedBufferType distributedType,
                                                                 mlir::AffineMap memPerm,
                                                                 mlir::PatternRewriter& rewriter, int64_t portCount,
                                                                 Logger logger);

    static mlir::LogicalResult unrollDuplicatedOutput(VPUIP::PermuteDMAOp permuteOp,
                                                      VPUIP::DistributedBufferType distributedType,
                                                      mlir::AffineMap memPerm, mlir::PatternRewriter& rewriter, int64_t,
                                                      Logger logger);

    static mlir::LogicalResult unrollDuplicatedInputAndOutput(VPUIP::PermuteDMAOp permuteOp, mlir::AffineMap memPerm,
                                                              mlir::PatternRewriter& rewriter, int64_t, Logger logger);

    static mlir::LogicalResult unrollDuplicatedInput(VPUIP::PermuteDMAOp permuteOp, mlir::AffineMap memPerm,
                                                     mlir::PatternRewriter& rewriter, int64_t, Logger logger);
};

class UnrollSingleClusterPermuteDMA {
public:
    static mlir::LogicalResult unroll(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter, int64_t portCount,
                                      Logger logger);
};

template <typename UnrollSingleCluster, typename UnrollMultiCluster>
mlir::LogicalResult unrollPermuteDMA(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter, int64_t portCount,
                                     Logger logger) {
    // Skip PermuteDMA ops which have been unrolled by checking mem_perm attribute
    if (permuteOp.getMemPermAttr() == nullptr) {
        return mlir::failure();
    }

    logger.trace("Permute rewriter operation '{0}' at '{1}'", permuteOp->getName(), permuteOp->getLoc());

    const auto input = permuteOp.getInput();
    const auto output = permuteOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());

    auto inDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType);
    auto outDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);

    // Dispatch between single cluster and multi-cluster tasks.
    //  - multi-cluster tasks have at least one distributed buffer and do not have DMADescriptorAttr
    //  - single cluster tasks either do not have any distributed buffers or have DMADescriptorAttr resulted
    //  from multi-cluster task unrolling
    //  - only form of single-cluster tasks with distributed buffers is with DUPLICATED output buffer
    if ((inDistributedType || outDistributedType) && !permuteOp.getDmaDescriptorAttr()) {
        // Unroll multi-cluster task
        logger.trace("process permute with DistributedType at {0}", permuteOp);

        VPUX_THROW_UNLESS(permuteOp.getMemPerm().has_value(),
                          "Can not get memPerm attribute from PermuteDMA layer at {0}", permuteOp.getLoc());
        const auto memPerm = permuteOp.getMemPerm().value();

        if (inDistributedType != nullptr && outDistributedType != nullptr) {
            return UnrollMultiCluster::unrollDuplicatedInputAndOutput(permuteOp, memPerm, rewriter, portCount, logger);
        } else if (inDistributedType != nullptr) {
            return UnrollMultiCluster::unrollDuplicatedInput(permuteOp, memPerm, rewriter, portCount, logger);
        }

        VPUX_THROW_UNLESS(inputType.getMemoryKind() == VPU::MemoryKind::DDR &&
                                  outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                          "Unexpected memory space. Got: input {0}, output {1}", inputType.getMemoryKind(),
                          outputType.getMemoryKind());

        VPUX_THROW_WHEN(outDistributedType == nullptr, "Expect distributed type for permute op output, actual: {0}",
                        outputType);

        VPUX_THROW_UNLESS(VPUIP::doesPermuteDMATileDimSupportWrapInCluster(inputType, outputType, memPerm,
                                                                           outDistributedType, logger),
                          "Unsupported PermuteDMA under cluster tiling at '{0}'", permuteOp->getLoc());

        const auto distributionAttr = outDistributedType.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
            return UnrollMultiCluster::unrollSegmentedOrOverlappedOutput(permuteOp, outDistributedType, memPerm,
                                                                         rewriter, portCount, logger);
        } else if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
                   VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
            return UnrollMultiCluster::unrollDuplicatedOutput(permuteOp, outDistributedType, memPerm, rewriter,
                                                              portCount, logger);
        } else {
            VPUX_THROW("Unsupported distributed mode");
        }
    } else {
        // Unroll single cluster task
        return UnrollSingleCluster::unroll(permuteOp, rewriter, portCount, logger);
    }

    return mlir::failure();
}

mlir::LogicalResult rewritePermuteDMA(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter, int64_t portCount,
                                      Logger logger);

}  // namespace vpux::arch37xx
