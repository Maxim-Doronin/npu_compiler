//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/TypeSwitch.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CREATENEWWEIGHTTABLESDATA
#define GEN_PASS_DEF_CREATENEWWEIGHTTABLESDATA
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

struct WeightTableKey {
    SmallVector<int32_t> data;
    mlir::Type weightsElemType;
    SmallVector<int32_t> workloadChannels;

    bool operator==(const WeightTableKey& other) const {
        return data == other.data && weightsElemType == other.weightsElemType &&
               workloadChannels == other.workloadChannels;
    }
};

}  // namespace

namespace llvm {
template <>
struct DenseMapInfo<WeightTableKey> {
    static inline WeightTableKey getEmptyKey() {
        return {{}, DenseMapInfo<mlir::Type>::getEmptyKey(), {}};
    }
    static inline WeightTableKey getTombstoneKey() {
        return {{}, DenseMapInfo<mlir::Type>::getTombstoneKey(), {}};
    }
    static unsigned getHashValue(const WeightTableKey& key) {
        auto hash = llvm::hash_combine_range(key.data.begin(), key.data.end());
        hash = llvm::hash_combine(hash, DenseMapInfo<mlir::Type>::getHashValue(key.weightsElemType));
        hash = llvm::hash_combine(hash,
                                  llvm::hash_combine_range(key.workloadChannels.begin(), key.workloadChannels.end()));
        return hash;
    }
    static bool isEqual(const WeightTableKey& lhs, const WeightTableKey& rhs) {
        return lhs == rhs;
    }
};
}  // namespace llvm

namespace {

SmallVector<int32_t> extractWorkloadChannels(VPU::NCEOpInterface nceOp) {
    auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();
    VPUX_THROW_UNLESS(!workloads.empty(), "No workloads were retrieved from '{0}' at '{1}'", nceOp->getName(),
                      nceOp->getLoc());

    SmallVector<int32_t> workloadChannels;
    for (auto workload : workloads) {
        auto outSizes = workload.getConstOutputSizes();
        workloadChannels.push_back(outSizes[Dims4D::Act::C.ind()]);
    }

    // If the weight table has DUPLICATED distribution type, use a single channel count, as workload sizes are
    // equal and we can reuse the same table for each workload.
    const auto weightTableOperand = nceOp.getWeightZeroPointsOperand() ? nceOp.getWeightZeroPointsOperand()
                                                                       : nceOp.getWeightTableDataPtrOperand();
    VPUX_THROW_UNLESS(weightTableOperand != nullptr, "Can't get weight table operand for '{0}' at '{1}'",
                      nceOp->getName(), nceOp->getLoc());

    const auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(weightTableOperand.getType());
    if (distributedType != nullptr &&
        distributedType.getDistribution().getMode().getValue() == VPU::DistributionMode::DUPLICATED) {
        return {workloadChannels[0]};
    }

    return workloadChannels;
}

template <typename WeightTableType>
WeightTableType updateWeightTableOp(mlir::IRRewriter& rewriter, VPU::NCEOpInterface nceOp,
                                    WeightTableType weightTableOp, ArrayRef<int32_t> workloadChannels,
                                    llvm::DenseMap<WeightTableKey, mlir::Operation*>& createdTables, Logger log) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    const auto outChannels = outputType.getShape()[Dims4D::Act::C];

    // Create the weight table data with correct workload sizes and weights element type information.
    auto weightsElemType = mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType()).getElementType();

    SmallVector<int32_t> weightTableData;
    mlir::RankedTensorType newOutputType;

    // Create the table data based on the weight table type.
    if (mlir::isa<VPU::ZeroPointTableOp>(weightTableOp)) {
        weightTableData = VPU::materializeZeroPointTable(weightsElemType, outChannels, workloadChannels);

        const auto zeroPointDataShape = VPU::NCESparsity::inferWeightsTableShape(
                static_cast<int64_t>(weightTableData.size()), /*newFormat=*/true);
        newOutputType = mlir::RankedTensorType::get(zeroPointDataShape.raw(), rewriter.getI8Type());

    } else if (mlir::isa<VPU::DataPointerTableOp>(weightTableOp)) {
        // Default to 1 cluster if there's no distribution
        int64_t numClusters = 1;
        if (auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(nceOp->getResult(0).getType())) {
            numClusters = distributedType.getDistribution().getNumClusters().getInt();
        }

        weightTableData = VPU::materializeDataPointerTable(rewriter.getContext(), workloadChannels,
                                                           nceOp.getWeightsOperand(), 0, outChannels, numClusters);

        const auto dataPointerDataShape = VPU::NCESparsity::inferWeightsTableShape(
                static_cast<int64_t>(weightTableData.size()), /*newFormat=*/true);
        newOutputType = mlir::RankedTensorType::get(dataPointerDataShape.raw(), getSInt32Type(rewriter.getContext()));
    } else {
        VPUX_THROW("Unsupported weight table op '{0}' at '{1}'", weightTableOp->getName(), weightTableOp->getLoc());
    }

    // Check if we already created a table with this exact data and attributes
    WeightTableKey key{weightTableData, weightsElemType,
                       SmallVector<int32_t>(workloadChannels.begin(), workloadChannels.end())};
    auto it = createdTables.find(key);
    if (it != createdTables.end()) {
        log.trace("Reusing previously created weight table with matching data and attributes");
        return mlir::cast<WeightTableType>(it->second);
    }

    rewriter.setInsertionPoint(weightTableOp);
    auto newWeightTableOp = rewriter.create<WeightTableType>(weightTableOp->getLoc(), newOutputType,
                                                             mlir::TypeAttr::get(weightsElemType),
                                                             getIntArrayAttr(rewriter.getContext(), workloadChannels),
                                                             getIntArrayAttr(rewriter.getContext(), weightTableData));

    // Track this newly created table
    createdTables[key] = newWeightTableOp;

    log.trace("Updated weights table: {0}", newWeightTableOp);
    return newWeightTableOp;
}

template <typename WeightTableType>
void updateCopyOp(mlir::IRRewriter& rewriter, VPU::NCEOpInterface nceOp, VPU::CopyOp oldCopyOp,
                  WeightTableType newWeightTableOp, Logger log) {
    auto oldOutputType = oldCopyOp.getOutput().getType();

    auto oldDistType = mlir::dyn_cast<VPU::DistributedTensorType>(oldOutputType);
    if (oldDistType == nullptr) {
        // No distribution, just update CopyOp with corrected input/output sizes and its attributes
        auto newWeightTableType = mlir::cast<vpux::NDTypeInterface>(newWeightTableOp.getOutput().getType());
        auto oldNDType = mlir::cast<vpux::NDTypeInterface>(oldOutputType);

        auto newOutputType =
                newWeightTableType
                        .changeShapeElemType(newWeightTableType.getShape(), newWeightTableType.getElementType())
                        .changeMemSpace(oldNDType.getMemSpace())
                        .changeDimsOrder(oldNDType.getDimsOrder());

        rewriter.setInsertionPoint(oldCopyOp);
        auto newCopyOp = rewriter.replaceOpWithNewOp<VPU::CopyOp>(
                oldCopyOp, newOutputType, newWeightTableOp.getOutput(), oldCopyOp.getOutMemSpaceAttr());
        log.trace("Updated CopyOp for weight table: {0}", newCopyOp);
        return;
    }

    auto oldDistribution = oldDistType.getDistribution();

    auto oldNumTiles = oldDistribution.getNumTiles();
    auto newNumTiles = oldNumTiles ? parseIntArrayAttr<int64_t>(oldNumTiles) : SmallVector<int64_t>{};

    auto oldAlignment = oldDistribution.getAlignment();
    auto newAlignment = oldAlignment ? parseIntArrayAttr<int64_t>(oldAlignment) : SmallVector<int64_t>{};

    auto newWeightTableType = mlir::cast<vpux::NDTypeInterface>(newWeightTableOp.getOutput().getType());

    bool isZeroPoint4Bit = false;
    if (mlir::isa<VPU::ZeroPointTableOp>(newWeightTableOp)) {
        auto newWeightsQuantPerAxisType =
                mlir::cast<mlir::quant::UniformQuantizedPerAxisType>(newWeightTableOp.getWeightsElemType());
        // MLIR quantization type system guarantees that zero points are of the storage type (see
        // mlir/include/mlir/Dialect/Quant/IR/QuantBase.td)
        isZeroPoint4Bit = newWeightsQuantPerAxisType.getStorageTypeIntegralWidth() == 4;
    }

    auto newMemoryShapes = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getMemoryShapes());
    auto newComputeShapes = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getComputeShapes());

    auto newMemoryOffsets = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getMemoryOffsets());
    auto newComputeOffsets = parseIntArrayOfArrayAttr<int64_t>(oldDistribution.getComputeOffsets());

    bool isDuplicatedMode = oldDistribution.getMode().getValue() == VPU::DistributionMode::DUPLICATED;
    auto numClusters = oldDistribution.getNumClusters().getInt();

    auto getAlignedSize = [&](int64_t workloadSize) {
        if (mlir::isa<VPU::ZeroPointTableOp>(newWeightTableOp)) {
            return VPU::NCESparsity::NewWeightsTableFormatMapper::getZPTableAlignmentForWorkload(
                    isZeroPoint4Bit, static_cast<int32_t>(workloadSize));
        } else if (mlir::isa<VPU::DataPointerTableOp>(newWeightTableOp)) {
            return VPU::NCESparsity::NewWeightsTableFormatMapper::getNewPointerTableLogicalAlignmentForWorkload(
                    static_cast<int32_t>(workloadSize));
        } else {
            VPUX_THROW("Unsupported weight table op '{0}' at '{1}'", newWeightTableOp->getName(),
                       newWeightTableOp->getLoc());
        }
    };

    if (isDuplicatedMode) {
        // In DUPLICATED mode, each tile gets the same weight table
        for (int64_t i = 0; i < numClusters; i++) {
            int32_t alignedWorkloadSize = getAlignedSize(newMemoryShapes[i][0]);

            newMemoryShapes[i][0] = alignedWorkloadSize;
            newComputeShapes[i][0] = alignedWorkloadSize;
            // Offsets remain zero in duplicated mode
        }
    } else {
        // In SEGMENTED mode, each tile gets its own piece from one weight table based on workload channels
        int32_t cumulativeOffset = 0;
        for (int64_t i = 0; i < numClusters; i++) {
            int32_t alignedWorkloadSize = getAlignedSize(newMemoryShapes[i][0]);

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
            clusteredOp, newWeightTableType, oldDistribution.getMode().getValue(), newNumTiles,
            oldDistribution.getNumClusters().getInt(), newAlignment,
            oldDistribution.getUniformDistributedSegments() != nullptr, overlapParams, std::nullopt);

    rewriter.setInsertionPoint(oldCopyOp);
    auto newCopyOp = rewriter.replaceOpWithNewOp<VPU::CopyOp>(
            oldCopyOp, newDistributedType, newWeightTableOp.getOutput(), oldCopyOp.getOutMemSpaceAttr());

    log.trace("Updated CopyOp for weight table: {0}", newCopyOp);
}

mlir::Operation* findWeightTableOp(mlir::Value value) {
    auto parentOp = value.getDefiningOp();
    VPUX_THROW_WHEN(parentOp == nullptr, "Unexpected NCE parent operation");

    return llvm::TypeSwitch<mlir::Operation*, mlir::Operation*>(parentOp)
            .Case<VPU::ZeroPointTableOp, VPU::DataPointerTableOp>([](mlir::Operation* op) {
                return op;
            })
            .Case<VPU::CopyOp>([&](VPU::CopyOp copyOp) {
                return findWeightTableOp(copyOp.getInput());
            })
            .Case<VPU::SliceOp>([&](VPU::SliceOp sliceOp) {
                return findWeightTableOp(sliceOp.getInput());
            })
            .Default([](mlir::Operation* op) -> mlir::Operation* {
                VPUX_THROW("Unexpected operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
}

template <typename WeightTableOpType>
void processWeightTableOp(mlir::IRRewriter& rewriter, VPU::NCEOpInterface nceOp, mlir::Value tableOperand,
                          llvm::DenseMap<WeightTableKey, mlir::Operation*>& createdTables, Logger log) {
    auto oldWeightTableOp = mlir::cast<WeightTableOpType>(findWeightTableOp(tableOperand));

    const auto workloadChannels = extractWorkloadChannels(nceOp);

    auto newWeightTableOp =
            updateWeightTableOp(rewriter, nceOp, oldWeightTableOp, workloadChannels, createdTables, log);

    if (auto oldCopyOp = tableOperand.getDefiningOp<VPU::CopyOp>()) {
        updateCopyOp(rewriter, nceOp, oldCopyOp, newWeightTableOp, log);

        if (auto sliceOp = oldCopyOp.getInput().getDefiningOp<VPU::SliceOp>()) {
            rewriter.replaceOp(sliceOp, newWeightTableOp.getResult());
        }
    }

    if (oldWeightTableOp->use_empty()) {
        rewriter.eraseOp(oldWeightTableOp);
    }
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

    // Track created weight tables by their data and attributes to avoid duplicates
    llvm::DenseMap<WeightTableKey, mlir::Operation*> createdTables;

    // Process weight table operations connected to NCE operations
    func->walk([&](VPU::NCEOpInterface nceOp) {
        if (!VPU::MPEEngineConfig::useNewWeightTableFormat(nceOp, /*isCompressConv=*/false)) {
            return;
        }

        // If we find any table, it needs to be updated
        auto dataPointerTable = nceOp.getWeightTableDataPtrOperand();
        auto zeroPointTable = nceOp.getWeightZeroPointsOperand();

        if (dataPointerTable == nullptr && zeroPointTable == nullptr) {
            return;
        }

        if (dataPointerTable) {
            processWeightTableOp<VPU::DataPointerTableOp>(rewriter, nceOp, dataPointerTable, createdTables, _log);
        } else if (zeroPointTable) {
            processWeightTableOp<VPU::ZeroPointTableOp>(rewriter, nceOp, zeroPointTable, createdTables, _log);
        }
    });
}

}  // namespace

//
// createCreateNewWeightTablesDataPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCreateNewWeightTablesDataPass(Logger log) {
    return std::make_unique<CreateNewWeightTablesDataPass>(log);
}
