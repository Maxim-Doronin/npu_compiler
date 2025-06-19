//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// LayerOpInterface
//

mlir::LogicalResult vpux::VPU::verifyLayer(mlir::Operation* op) {
    if (VPU::verifyOpLayout(op).failed()) {
        return mlir::failure();
    }

    return IE::verifyLayer(op);
}

//
// SparseOpInterface
//

bool vpux::VPU::supportsSparseInputs(mlir::Operation* op) {
    const auto compressedInput = [](mlir::Value operand) {
        auto inputShape = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getShape();
        if (inputShape[Dims4D::Act::C] < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
            return true;
        }
        return false;
    };
    if (compressedInput(op->getOperand(0))) {
        return false;
    }
    if (mlir::isa<VPU::NCEEltwiseOp>(op) && compressedInput(op->getOperand(1))) {
        return false;
    }

    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContainsAny(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_INPUTS);
    }
    return false;
}

bool vpux::VPU::supportsSparseOutputs(mlir::Operation* op) {
    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContainsAny(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_OUTPUTS);
    }
    return false;
}

bool vpux::VPU::supportsSparseWeights(mlir::Operation* op) {
    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContainsAny(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_WEIGHTS);
    }
    return false;
}

bool vpux::VPU::supportsSparseData(mlir::Operation* op) {
    return supportsSparseInputs(op) && supportsSparseOutputs(op);
}

mlir::LogicalResult vpux::VPU::details::validateWorkloadsRegion(mlir::Location loc, mlir::Region& workloads) {
    for (auto& workloadOp : workloads.getOps()) {
        if (!mlir::isa<DPUWorkloadOp>(workloadOp)) {
            return errorAt(loc, "Got unsupported Operation '{0}' in 'workloads' region", workloadOp.getName());
        }
    }

    return mlir::success();
}

mlir::Operation* vpux::VPU::details::addWorkload(mlir::Region& workloads, mlir::OpBuilder& builder, mlir::Location loc,
                                                 ShapeRef offsets, ShapeRef sizes, PaddingAttr pad, MPEMode mpeMode,
                                                 mlir::IntegerAttr clusterId) {
    if (workloads.empty()) {
        workloads.emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&workloads.front());

    const auto offsetsAttr = getIntArrayAttr(builder.getContext(), offsets);
    const auto sizesAttr = getIntArrayAttr(builder.getContext(), sizes);

    return builder.create<DPUWorkloadOp>(loc, offsetsAttr, sizesAttr, pad, mpeMode, clusterId);
}

mlir::LogicalResult vpux::VPU::details::verifyInputTypeOp(mlir::Operation* op, vpux::NDTypeInterface inputType) {
    auto alignedOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op);
    if (alignedOp == nullptr) {
        return mlir::success();
    }

    // TODO: #123810 split verifier into arch-specific parts
    bool supportsInputActCompression = false;
    auto alignment = alignedOp.getInputChannelAlignment();
    if (mlir::isa<VPU::NCECompressConvolutionOp>(op)) {
        alignment = vpux::VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM;
        supportsInputActCompression = true;
    }

    return mlir::success(
            vpux::VPU::NCEInvariant::isInputActTypeSupported(inputType, alignment, supportsInputActCompression));
}

//
// TilingBuilderOpInterface
//

mlir::Value vpux::VPU::makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal,
                                const TileInfo& tile, StringRef valName) {
    if (tile.shape == getShape(origVal)) {
        return origVal;
    }

    const auto loc = appendLoc(baseLoc, "{0} tile {1}", valName, tile.offsets);

    auto sliceOp = builder.create<VPU::SliceOp>(loc, origVal, tile.offsets, tile.shape);
    return sliceOp.getResult();
}

//
// TilingInfoOpInterface
//

mlir::LogicalResult vpux::VPU::verifyTilingInfo(mlir::Operation* op) {
    if (!mlir::isa<VPU::TilingBuilderOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' provides TilingInfoOpInterface, but not TilingBuilderOpInterface",
                       op->getName());
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Unsupported operation '{0}', it must have one and only one result", op->getName());
    }

    return mlir::success();
}

//
// EltwiseOp
//

mlir::LogicalResult vpux::VPU::verifyEltwiseOp(mlir::Operation* op) {
    if (!mlir::isa<VPU::LayerOpInterface>(op)) {
        return errorAt(op, "EltwiseOp trait is applied to non layer operation");
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Operation with multiple results can't be EltwiseOp");
    }

    if (op->hasAttr("auto_broadcast")) {
        auto autoBroadcast = mlir::dyn_cast<vpux::IE::AutoBroadcastTypeAttr>(op->getAttr("auto_broadcast"));
        if (autoBroadcast == nullptr) {
            return errorAt(op, "Auto broadcast attribute cannot be cast");
        }
        auto broadcast = autoBroadcast.getValue();

        SmallVector<ArrayRef<int64_t>> inputShapes;
        for (auto operand : op->getOperands()) {
            const auto shape = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getShape().raw();
            inputShapes.push_back(shape);
        }

        const auto outputShape = IE::broadcastEltwiseShape(inputShapes, broadcast, op->getLoc());

        if (mlir::failed(outputShape)) {
            return errorAt(op, "Eltwise inputs cannot be broadcast");
        }
    }

    return mlir::success();
}

//
// NCEOpInterface
//

mlir::LogicalResult vpux::VPU::verifyNCEOp(mlir::Operation* op) {
    if (!mlir::isa<VPU::NCEOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' is not NCE", op->getName());
    }

    auto nceOp = mlir::cast<VPU::NCEOpInterface>(op);
    if (vpux::VPU::details::validateWorkloadsRegion(nceOp->getLoc(), nceOp.getWorkloads()).failed()) {
        return mlir::failure();
    }

    auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op);
    if (!iface) {
        return errorAt(op, "NCE Operation '{0}' must attach AlignedChannelsOpInterface", op->getName());
    }

    return iface.verifyChannels();
}

//
// ClusteredOpInterface
//

vpux::NDTypeInterface vpux::VPU::getDistributedTypeForOpOperand(mlir::Operation* op, mlir::OpOperand& operand,
                                                                bool hasExplicitDistributedAttr,
                                                                SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return nullptr;
    }

    if (mlir::isa<VPU::SWOpInterface>(op)) {
        return getSwDistributedTypeForOpOperand(clusteredOp, operand, siblingsAnalysis, hasExplicitDistributedAttr);
    }

    VPUX_THROW("Can't generate distributed-type for operand");
}

vpux::NDTypeInterface vpux::VPU::getDistributedTypeForOpResult(mlir::Operation* op, mlir::Value result,
                                                               VPU::MultiClusterStrategy strategy,
                                                               SiblingOpsAnalysis& siblingsAnalysis,
                                                               bool hasExplicitDistributedAttr) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return nullptr;
    }

    return getDistributedOutputTensorType(clusteredOp, mlir::cast<vpux::NDTypeInterface>(result.getType()),
                                          siblingsAnalysis, strategy, hasExplicitDistributedAttr);
}

bool vpux::VPU::supportSwOpLoweringAsDMA(mlir::Operation* op) {
    VPUX_THROW_UNLESS(mlir::isa_and_nonnull<VPU::SWOpInterface>(op), "Unexpected operation type");
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<VPU::ConvertOp>([](auto convertOp) {
                return isConvertSupportedOnDMA<VPU::ConvertOp>(convertOp);
            })
            .Default([](mlir::Operation*) {
                return false;
            });
}

//
// isPureViewLike
//

bool vpux::VPU::isPureViewOp(mlir::Operation* op) {
    return mlir::isa<VPU::ViewLikeOpInterface, mlir::ViewLikeOpInterface, vpux::MultiViewOpInterface,
                     vpux::GroupedViewOpInterface, VPU::GroupedViewLikeOpInterface>(op);
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/ops_interfaces.cpp.inc>
