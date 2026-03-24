//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_RELOCATEWEIGHTTABLEFORREUSE
#define GEN_PASS_DEF_RELOCATEWEIGHTTABLEFORREUSE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

std::tuple<SmallVector<int64_t>, SmallVector<uint32_t>> getOffsetsAndWeightsPtrsForConv(
        VPU::DistributedTensorType distrType, bool isDistrType) {
    if (!isDistrType) {
        return {SmallVector<int64_t>(1, 0), SmallVector<uint32_t>(1, 0)};
    }

    SmallVector<int64_t> offsets;
    size_t numClusters = 1;
    numClusters = distrType.getDistribution().getNumClusters().getInt();
    const auto perClusterShapeOffsets = distrType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Mismatch between the number of shape offsets '{0}' and the number of clusters '{1}'.",
                      perClusterShapeOffsets.size(), numClusters);
    for (auto clusterOffsets : perClusterShapeOffsets | indexed) {
        offsets.push_back(clusterOffsets.value()[Dims4D::Filter::OC]);
    }
    SmallVector<uint32_t> weightsPtrPerCluster(numClusters, static_cast<int32_t>(0));
    return {offsets, weightsPtrPerCluster};
}

std::tuple<SmallVector<int64_t>, SmallVector<uint32_t>> getOffsetsAndWeightsPtrsForMatMul(vpux::NDTypeInterface type) {
    SmallVector<int64_t> offsets;
    auto shape = type.getShape();
    for (auto group : irange(shape[DimsGroups5D::Filter::G])) {
        offsets.push_back(group * shape[DimsGroups5D::Filter::OC]);
    }
    SmallVector<uint32_t> weightsPtrPerCluster(shape[DimsGroups5D::Filter::G], static_cast<int32_t>(0));
    return {offsets, weightsPtrPerCluster};
}

std::tuple<SmallVector<int64_t>, SmallVector<uint32_t>> getOffsetsAndWeightsPtrsForSCF(
        mlir::scf::ForallOp forallOp, mlir::tensor::ExtractSliceOp constSourceSliceOp) {
    auto lowerBounds = forallOp.getMixedLowerBound();
    auto upperBounds = forallOp.getMixedUpperBound();
    auto steps = forallOp.getMixedStep();

    // guard against multi-dimensional or empty scf.forall
    if (lowerBounds.size() != 1 || upperBounds.size() != 1 || steps.size() != 1) {
        return {SmallVector<int64_t>{}, SmallVector<uint32_t>{}};
    }

    auto lb = mlir::getConstantIntValue(lowerBounds[0]);
    auto ub = mlir::getConstantIntValue(upperBounds[0]);
    auto step = mlir::getConstantIntValue(steps[0]);

    VPUX_THROW_UNLESS(lb.has_value() && ub.has_value() && step.has_value(),
                      "Expected constant bounds for scf.forall loop");

    SmallVector<int64_t> forallOffsets;
    for (int64_t i = lb.value(); i < ub.value(); i += step.value()) {
        forallOffsets.push_back(i);
    }

    // Check if there's a parent scf.for that tiles the weight table.
    // Instead of using getParentOfType (which might pick a wrong ForOp in deeply nested
    // loops), trace constSourceSliceOp's dynamic offsets to find the ForOp whose induction
    // variable actually feeds the weight table slice.
    mlir::scf::ForOp tilingForOp = nullptr;
    for (auto offset : constSourceSliceOp.getMixedOffsets()) {
        if (mlir::getConstantIntValue(offset).has_value()) {
            continue;
        }
        auto val = mlir::cast<mlir::Value>(offset);
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val);
        if (!blockArg) {
            continue;
        }
        auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp());
        if (forOp && blockArg == forOp.getInductionVar()) {
            tilingForOp = forOp;
            break;
        }
    }

    SmallVector<int64_t> offsets;
    if (tilingForOp != nullptr) {
        // Nested scf.for (tiling) + scf.forall (multiclustering): combine offsets.
        // The scf.for splits the weight table into tiles (e.g. 0..32, 32..64), and
        // scf.forall splits each tile across clusters (e.g. 0..16, 16..32).
        // Full offsets: for each tile start, add each cluster offset.
        auto forLb = mlir::getConstantIntValue(tilingForOp.getLowerBound());
        auto forUb = mlir::getConstantIntValue(tilingForOp.getUpperBound());
        auto forStep = mlir::getConstantIntValue(tilingForOp.getStep());
        if (forLb.has_value() && forUb.has_value() && forStep.has_value()) {
            for (int64_t tile = forLb.value(); tile < forUb.value(); tile += forStep.value()) {
                for (auto clusterOffset : forallOffsets) {
                    offsets.push_back(tile + clusterOffset);
                }
            }
        } else {
            offsets = std::move(forallOffsets);
        }
    } else {
        offsets = std::move(forallOffsets);
    }

    SmallVector<uint32_t> weightsPtrPerCluster(offsets.size(), static_cast<uint32_t>(0));
    return {offsets, weightsPtrPerCluster};
}

class RelocateWeightTableForReusePass final :
        public VPU::impl::RelocateWeightTableForReuseBase<RelocateWeightTableForReusePass> {
public:
    explicit RelocateWeightTableForReusePass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RelocateWeightTableForReusePass::safeRunOnFunc() {
    auto func = getOperation();

    // If neither weights-table-reuse is enabled nor the function is a pure vertical fusion region,
    // skip the relocation of weights table for reuse.
    if (!config::isWeightsTableReuseEnabled(func)) {
        _log.trace("Skipping relocation of weights table for reuse because the function is not supported {0}",
                   func->getLoc());
        return;
    }

    func.walk([&](VPU::NCEOpInterface nceOp) {
        if (!mlir::isa<VPU::NCEMatMulOp, VPU::NCEConvolutionOp>(nceOp)) {
            return;
        }

        // For new weights table format which actually don't have weights table, we can not apply the optimization
        if (VPU::MPEEngineConfig::useNewWeightTableFormat(nceOp, /*isCompressConv*/ false)) {
            return;
        }

        _log.trace("[{0}]: NCE operation at {1}", nceOp->getName(), nceOp->getLoc());
        Const::DeclareOp cstOp = nullptr;
        const auto weightsTable = nceOp->getOperand(2);
        auto unrolledOp = mlir::dyn_cast<VPU::UnrolledTypeOp>(weightsTable.getDefiningOp());
        auto extractSliceOp = mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(weightsTable.getDefiningOp());

        mlir::tensor::ExtractSliceOp constSourceSliceOp = extractSliceOp;

        if (unrolledOp != nullptr) {
            cstOp = mlir::dyn_cast<Const::DeclareOp>(unrolledOp.getInput().getDefiningOp());
        } else if (extractSliceOp != nullptr) {
            // Walk up the chain of extract_slice ops to find the const.Declare
            auto source = extractSliceOp.getSource();
            while (auto parentSlice = mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(source.getDefiningOp())) {
                constSourceSliceOp = parentSlice;
                source = parentSlice.getSource();
            }
            cstOp = mlir::dyn_cast<Const::DeclareOp>(source.getDefiningOp());
        } else {
            cstOp = mlir::dyn_cast<Const::DeclareOp>(weightsTable.getDefiningOp());
        }

        if (cstOp == nullptr) {
            return;
        }

        const auto weightTableDistrType = mlir::dyn_cast<VPU::DistributedTensorType>(weightsTable.getType());
        const bool isDistrType = unrolledOp != nullptr && weightTableDistrType != nullptr;
        auto forallOp = nceOp->getParentOfType<mlir::scf::ForallOp>();
        const bool isSCF = forallOp != nullptr && extractSliceOp != nullptr;
        const auto weights = nceOp->getOperand(1);
        if (mlir::isa<vpux::VPU::SparseTensorType>(weights.getType())) {
            return;
        }

        const auto channelOffset = 0;
        // For SCF flow, use the full constant type (not the sliced type)
        auto weightTableType =
                mlir::cast<vpux::NDTypeInterface>(isSCF ? cstOp.getOutput().getType() : weightsTable.getType());
        auto shapeTotalSize = weightTableType.getShape().totalSize();
        auto elementSize = weightTableType.getElemTypeSize().count() / CHAR_BIT;
        auto weightsElemBitSize = getElemTypeSize(weights.getType()).count();

        auto isMatMul = mlir::isa<VPU::NCEMatMulOp>(nceOp);
        auto [offsets, weightsPtrPerCluster] =
                isMatMul ? getOffsetsAndWeightsPtrsForMatMul(weightTableType)
                : isSCF  ? getOffsetsAndWeightsPtrsForSCF(forallOp, constSourceSliceOp)
                         : getOffsetsAndWeightsPtrsForConv(weightTableDistrType, isDistrType);

        auto originalOC = 0;
        if (VPU::canAutopadOutput(nceOp.getOperation())) {
            originalOC = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()).getShape()[Dims4D::Act::C];
        }

        auto newConstAttr = cstOp.getContentAttr()
                                    .transform()
                                    .relocateWeightsTablePointers(
                                            weightsPtrPerCluster, VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY,
                                            ShapeRef(offsets), (shapeTotalSize * elementSize), weightsElemBitSize,
                                            nullptr, channelOffset, originalOC)
                                    .get();

        mlir::OpBuilder builder(cstOp);
        auto newConstOp =
                builder.create<Const::DeclareOp>(cstOp.getLoc(), cstOp.getOutput().getType(), std::move(newConstAttr));
        vpux::Const::foldSingleConstant(newConstOp);

        if (isDistrType) {
            unrolledOp->setOperand(0, newConstOp.getOutput());
        } else if (isSCF) {
            // Wire into the extract_slice that directly reads from the constant
            constSourceSliceOp.getSourceMutable().assign(newConstOp.getOutput());
        } else {
            nceOp->setOperand(2, newConstOp.getOutput());
        }
        if (cstOp->getUses().empty()) {
            cstOp.erase();
        }
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createRelocateWeightTableForReusePass(Logger log) {
    return std::make_unique<RelocateWeightTableForReusePass>(std::move(log));
}
