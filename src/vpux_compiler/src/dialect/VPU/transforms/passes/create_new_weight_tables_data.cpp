//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CREATENEWWEIGHTTABLESDATA
#define GEN_PASS_DEF_CREATENEWWEIGHTTABLESDATA
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

mlir::Operation* findZeroPointTableOp(mlir::Value value) {
    auto parentOp = value.getDefiningOp();
    VPUX_THROW_WHEN(parentOp == nullptr, "Unexpected NCE parent operation");

    return llvm::TypeSwitch<mlir::Operation*, mlir::Operation*>(parentOp)
            .Case<VPU::ZeroPointTableOp>([](VPU::ZeroPointTableOp op) {
                return op;
            })
            .Case<VPU::CopyOp>([&](VPU::CopyOp copyOp) {
                return findZeroPointTableOp(copyOp.getInput());
            })
            .Case<VPU::SliceOp>([&](VPU::SliceOp sliceOp) {
                return findZeroPointTableOp(sliceOp.getInput());
            })
            .Case<MultiViewOpInterface>([&](MultiViewOpInterface viewOp) {
                auto opResult = mlir::dyn_cast<mlir::OpResult>(value);
                VPUX_THROW_WHEN(opResult == nullptr, "Value '{0}' cannot be converted to an op result", value);

                const auto source = viewOp.getViewSource(opResult.getResultNumber());
                return findZeroPointTableOp(source);
            })
            .Default([](mlir::Operation* op) -> mlir::Operation* {
                VPUX_THROW("Unexpected operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
}

SmallVector<int64_t> extractWorkloadChannels(VPU::NCEOpInterface nceOp) {
    auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();
    VPUX_THROW_UNLESS(!workloads.empty(), "No workloads were retrieved from '{0}' at '{1}'", nceOp->getName(),
                      nceOp->getLoc());

    SmallVector<int64_t> workloadChannels;
    for (auto workload : workloads) {
        auto outSizes = workload.getConstOutputSizes();
        workloadChannels.push_back(outSizes[Dims4D::Act::C.ind()]);
    }

    // If the zero-point table has DUPLICATED distribution type, use a single channel count, as workload sizes are
    // equal and we can reuse the same table for each workload.
    auto mode = mlir::cast<VPU::DistributedTensorType>(nceOp.getWeightZeroPointsOperand().getType())
                        .getDistribution()
                        .getMode();
    if (mode.getValue() == VPU::DistributionMode::DUPLICATED) {
        return {workloadChannels[0]};
    }

    return workloadChannels;
}

VPU::ZeroPointTableOp updateZeroPointTableOp(mlir::IRRewriter& rewriter, VPU::ZeroPointTableOp zeroPointTableOp,
                                             VPU::NCEOpInterface nceOp, ArrayRef<int64_t> workloadChannels,
                                             Logger log) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    const auto outChannels = outputType.getShape()[Dims4D::Act::C];

    auto oldWeightsQuantPerAxisType =
            mlir::cast<mlir::quant::UniformQuantizedPerAxisType>(zeroPointTableOp.getWeightsElemType());
    int64_t startChannel = 0;

    // Find an offset from a SliceOp on the ZeroPointTableOp
    VPU::SliceOp sliceOp = nullptr;
    if (auto copyOp = nceOp.getWeightZeroPointsOperand().getDefiningOp<VPU::CopyOp>()) {
        if ((sliceOp = copyOp.getInput().getDefiningOp<VPU::SliceOp>())) {
            auto offsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            startChannel = offsets[Dims4D::Filter::OC.ind()];
        }
    }

    auto origZeroPoints = oldWeightsQuantPerAxisType.getZeroPoints();
    auto origScales = oldWeightsQuantPerAxisType.getScales();

    SmallVector<int64_t> tiledZeroPoints(origZeroPoints.begin() + startChannel,
                                         origZeroPoints.begin() + startChannel + outChannels);
    SmallVector<double> tiledScales(origScales.begin() + startChannel, origScales.begin() + startChannel + outChannels);

    // Create new quantized per axis type with corrected zero points and scales
    auto newWeightsQuantPerAxisType = mlir::quant::UniformQuantizedPerAxisType::get(
            oldWeightsQuantPerAxisType.getFlags(), oldWeightsQuantPerAxisType.getStorageType(),
            oldWeightsQuantPerAxisType.getExpressedType(), tiledScales, tiledZeroPoints,
            oldWeightsQuantPerAxisType.getQuantizedDimension(), oldWeightsQuantPerAxisType.getStorageTypeMin(),
            oldWeightsQuantPerAxisType.getStorageTypeMax());

    // Create the zero-point table data with correct workload sizes and weights element type information.
    const auto zeroPointData =
            VPU::materializeZeroPointTable(newWeightsQuantPerAxisType, outChannels, workloadChannels);

    const auto zeroPointDataShape =
            VPU::NCESparsity::inferWeightsTableShape(static_cast<int64_t>(zeroPointData.size()), /*newFormat=*/true);
    auto newOutputType = mlir::RankedTensorType::get(zeroPointDataShape.raw(), rewriter.getI8Type());

    rewriter.setInsertionPoint(zeroPointTableOp);
    auto newCreateZpTableOp = rewriter.create<VPU::ZeroPointTableOp>(
            zeroPointTableOp->getLoc(), newOutputType, mlir::TypeAttr::get(newWeightsQuantPerAxisType),
            getIntArrayAttr(rewriter.getContext(), workloadChannels),
            getIntArrayAttr(rewriter.getContext(), zeroPointData));

    if (sliceOp) {
        // If old ZeroPointTableOp was not sliced, no need to update it. Otherwise each SliceOp will be
        // replaced by ZeroPointTableOp with correct sizes.
        rewriter.replaceOp(sliceOp, newCreateZpTableOp.getResult());
    }

    log.trace("Updated ZeroPointTableOp: {0}", newCreateZpTableOp);
    return newCreateZpTableOp;
}

void updateCopyOp(mlir::IRRewriter& rewriter, VPU::CopyOp oldCopyOp, VPU::ZeroPointTableOp newCreateZpTableOp,
                  VPU::NCEOpInterface nceOp, Logger log) {
    auto oldOutputType = oldCopyOp.getOutput().getType();

    auto oldDistType = mlir::dyn_cast<VPU::DistributedTensorType>(oldOutputType);
    auto oldDistribution = oldDistType.getDistribution();

    auto oldNumTiles = oldDistribution.getNumTiles();
    auto newNumTiles = oldNumTiles ? parseIntArrayAttr<int64_t>(oldNumTiles) : SmallVector<int64_t>{};

    auto oldAlignment = oldDistribution.getAlignment();
    auto newAlignment = oldAlignment ? parseIntArrayAttr<int64_t>(oldAlignment) : SmallVector<int64_t>{};

    auto newZeroPointTableType = mlir::cast<vpux::NDTypeInterface>(newCreateZpTableOp.getOutput().getType());

    auto newWeightsQuantPerAxisType =
            mlir::cast<mlir::quant::UniformQuantizedPerAxisType>(newCreateZpTableOp.getWeightsElemType());
    // MLIR quantization type system guarantees that zero points are of the storage type (see
    // mlir/include/mlir/Dialect/Quant/IR/QuantBase.td)
    bool isZeroPoint4Bit = newWeightsQuantPerAxisType.getStorageTypeIntegralWidth() == 4;

    auto newMemoryShapes = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getMemoryShapes());
    auto newComputeShapes = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getComputeShapes());

    auto newMemoryOffsets = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getMemoryOffsets());
    auto newComputeOffsets = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getComputeOffsets());

    bool isDuplicatedMode = oldDistribution.getMode().getValue() == VPU::DistributionMode::DUPLICATED;
    auto numClusters = oldDistribution.getNumClusters().getInt();

    if (isDuplicatedMode) {
        // In DUPLICATED mode, each tile gets the same zero-point table
        for (int64_t i = 0; i < numClusters; i++) {
            int32_t alignedWorkloadSize = VPU::NCESparsity::NewWeightsTableFormatMapper::getZPTableAlignmentForWorkload(
                    isZeroPoint4Bit, static_cast<int32_t>(newMemoryShapes[i][0]));
            newMemoryShapes[i][0] = alignedWorkloadSize;
            newComputeShapes[i][0] = alignedWorkloadSize;
            // Offsets remain zero in duplicated mode
        }
    } else {
        // In SEGMENTED mode, each tile gets its own piece from one zero-point table based on workload channels
        int32_t cumulativeOffset = 0;
        for (int64_t i = 0; i < numClusters; i++) {
            int32_t alignedWorkloadSize = VPU::NCESparsity::NewWeightsTableFormatMapper::getZPTableAlignmentForWorkload(
                    isZeroPoint4Bit, static_cast<int32_t>(newMemoryShapes[i][0]));

            newMemoryShapes[i][0] = alignedWorkloadSize;
            newMemoryOffsets[i][0] = cumulativeOffset;

            // If zero-point is 4-bit, size of memory shapes/offsets might become smaller than compute
            // shapes/offsets, due to packing logic. So we need to update compute shapes/offsets accordingly.
            if (isZeroPoint4Bit) {
                newComputeShapes[i][0] = alignedWorkloadSize;
                newComputeOffsets[i][0] = cumulativeOffset;
            }

            cumulativeOffset += alignedWorkloadSize;
        }
    }

    auto overlapParams =
            VPU::OverlapDistributionParams(newMemoryShapes, newMemoryOffsets, newComputeShapes, newComputeOffsets);

    // Create new distributed type with manually set distribution parameters
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    auto newDistributedType = VPU::createExplicitDistributedTensorType(
            clusteredOp, newZeroPointTableType, oldDistribution.getMode().getValue(), newNumTiles,
            oldDistribution.getNumClusters().getInt(), newAlignment,
            oldDistribution.getUniformDistributedSegments() != nullptr, overlapParams, std::nullopt);

    rewriter.setInsertionPoint(oldCopyOp);
    auto newCopyOp = rewriter.replaceOpWithNewOp<VPU::CopyOp>(
            oldCopyOp, newDistributedType, newCreateZpTableOp.getOutput(), oldCopyOp.getOutMemSpaceAttr());

    log.trace("Updated CopyOp for ZeroPointTableOp: {0}", newCopyOp);
}

//
// CreateNewWeightTablesData
//

class CreateNewWeightTablesDataPass final :
        public VPU::impl::CreateNewWeightTablesDataBase<CreateNewWeightTablesDataPass> {
public:
    explicit CreateNewWeightTablesDataPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void CreateNewWeightTablesDataPass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::IRRewriter rewriter(&getContext());

    // Process ZeroPointTable operations connected to NCE operations
    llvm::SetVector<VPU::ZeroPointTableOp> oldCreateZpTableOps;
    func->walk([&](VPU::NCEOpInterface nceOp) {
        if (!VPU::MPEEngineConfig::useNewWeightTableFormat(nceOp, /*isCompressConv=*/false)) {
            return;
        }

        // Check if this operation has a zero-point table that needs updating.
        auto zeroPointTable = nceOp.getWeightZeroPointsOperand();
        if (zeroPointTable == nullptr) {
            return;
        }

        // Find ZeroPointTableOp through view operations and CopyOp.
        VPU::ZeroPointTableOp oldCreateZpTableOp =
                mlir::cast<VPU::ZeroPointTableOp>(findZeroPointTableOp(zeroPointTable));
        oldCreateZpTableOps.insert(oldCreateZpTableOp);

        const auto workloadChannels = extractWorkloadChannels(nceOp);
        auto newCreateZpTableOp = updateZeroPointTableOp(rewriter, oldCreateZpTableOp, nceOp, workloadChannels, _log);

        if (const auto oldCopyOp = zeroPointTable.getDefiningOp<VPU::CopyOp>()) {
            updateCopyOp(rewriter, oldCopyOp, newCreateZpTableOp, nceOp, _log);
        }
    });

    for (auto op : oldCreateZpTableOps) {
        if (op->use_empty()) {
            rewriter.eraseOp(op);
        }
    }
}

}  // namespace

//
// createCreateNewWeightTablesDataPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCreateNewWeightTablesDataPass(Logger log) {
    return std::make_unique<CreateNewWeightTablesDataPass>(log);
}
