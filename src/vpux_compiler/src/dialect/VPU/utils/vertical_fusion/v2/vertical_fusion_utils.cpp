//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

namespace vpux::VPU::VF::v2 {

bool isCmxOperation(mlir::Operation* operation, const bool checkTilingType) {
    if (!mlir::isa_and_nonnull<VPU::TilingInfoOpInterface, VPU::VerticalFusionOp>(operation)) {
        return false;
    }

    if (!operation->hasAttr(tilingStrategy)) {
        return true;
    }

    auto tiling = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy)));
    auto hasTiling = llvm::any_of(tiling, [](auto value) {
        return value > 1;
    });

    if (!hasTiling) {
        return true;
    }

    if (checkTilingType) {
        if (isSpatialTiling(tiling)) {
            return false;
        }

        const auto checkNCEFunc = [](mlir::Operation* oper) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(oper);
            auto hasWeights = nceOp != nullptr && nceOp.getWeightsOperand() != nullptr;
            auto needFullInput = true;
            if (auto vfInterface = mlir::dyn_cast<VPU::VerticalFusionOpInterface>(oper)) {
                auto restrictedAxes = vfInterface.restrictedFusionAxes();
                needFullInput =
                        !restrictedAxes.empty() && llvm::find(restrictedAxes, Dims4D::Act::C) != restrictedAxes.end();
            }
            return hasWeights && needFullInput;
        };

        if (auto vfUser = mlir::dyn_cast<VPU::VerticalFusionOp>(operation)) {
            auto userConfig = VFConfig(vfUser);
            return userConfig.getOperationsForTiling().size() == 1 &&
                   llvm::all_of(userConfig.getInputs(), checkNCEFunc);
        }
        return checkNCEFunc(operation);
    }

    const auto outputSize = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getTotalAllocSize();

    if (outputSize > VPU::getTotalCMXSize(operation)) {
        return false;
    }
    return !isSpatialTiling(tiling);
}

bool hasBeforeDDRUsers(mlir::Operation* prevOp, mlir::Operation* nextOp) {
    // check if previous operation has more than 1 users apart from nextOp
    // and all of them are in DDR
    auto uses = findUses(prevOp);
    if (uses.size() == 1) {
        return false;
    }

    const auto checkUser = [&](auto* use) {
        auto* user = use->getOwner();
        return user != nextOp && user->isBeforeInBlock(nextOp) && !v2::isCmxOperation(use->getOwner(), true);
    };

    return llvm::any_of(uses, checkUser);
}

namespace {
bool isDataTiledOnSameAxisWithMCStrategy(VPU::DistributedTensorType dataType, ArrayRef<int64_t> tiling) {
    if (dataType == nullptr) {
        return false;
    }
    auto mode = dataType.getDistribution().getMode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    auto tilingScheme = parseIntArrayAttr<int64_t>(dataType.getDistribution().getNumTiles());
    VPUX_THROW_WHEN(tilingScheme.size() != tiling.size(), "Unmatched tiling scheme and tiling size");
    auto axis = getDistributedTilingAxis(tilingScheme);
    VPUX_THROW_UNLESS(checked_cast<size_t>(axis) < tilingScheme.size(), "Invalid tiling axis");
    return tiling[axis] != 1;
}
}  // namespace

bool hasOutputSpilledForDifferentDataSizeUses(mlir::Operation* op) {
    auto outElementSize = getShape(op->getResult(0)).totalSize();
    auto usedBySizeChangedViewOps = llvm::all_of(op->getUsers(), [&](auto user) {
        if (!isPureViewOp(user)) {
            return false;
        }
        auto elementSize = getShape(user->getResult(0)).totalSize();
        return outElementSize != elementSize;
    });
    return usedBySizeChangedViewOps;
}

bool outputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op) {
    if (!isOpTiled(op)) {
        return false;
    }
    const auto tilingDim = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(vpux::tilingStrategy)));
    auto distributedType = mlir::dyn_cast_or_null<VPU::DistributedTensorType>(getDistributedOutputType(op));
    return isDataTiledOnSameAxisWithMCStrategy(distributedType, tilingDim);
}

bool inputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op, mlir::Value operand) {
    if (!isOpTiled(op)) {
        return false;
    }
    const auto tilingDim = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(vpux::tilingStrategy)));
    auto distributedType = mlir::dyn_cast_or_null<VPU::DistributedTensorType>(getDistributedInputType(op, operand));
    return isDataTiledOnSameAxisWithMCStrategy(distributedType, tilingDim);
}

}  // namespace vpux::VPU::VF::v2
