//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_convert_to_dpu_checker.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

using namespace vpux::IE::arch37xx;

//
// FuseConvertToDPUChecker
//

bool FuseConvertToDPUChecker::isFusionToParentDPUOpSupported(mlir::Operation* dpuOp, Logger log) const {
    auto parentInputType = mlir::cast<NDTypeInterface>(dpuOp->getOperand(0).getType());
    if (!mlir::isa<mlir::FloatType>(parentInputType.getElementType())) {
        log.trace("Parent input is not a float type = {0}.", parentInputType.getElementType());
        return false;
    }

    // For MaxPool, DPU SCL outputs FP16, not F32, so just setting bypass to conversion to F16 in PPE will not be enough
    // to get a proper FP32 output
    if (mlir::isa<IE::MaxPoolOp>(dpuOp)) {
        log.trace("Parent op of type {0} at loc {1} does not support FP32 output.", dpuOp->getName(), dpuOp->getLoc());
        return false;
    }

    // For Interp with MemPerm after, fusing FP16->FP32 Convert with Interp can lead to performance regression from
    // MemPerm. This is because the MemPerm needs to move 2x data than DPU solution with FP16 output, though the
    // Interp+Convert fusion is beneficial. So we don't fuse convert to DPU here. Experimental data: E#162186
    if (mlir::isa<IE::InterpolateOp>(dpuOp)) {
        log.trace("Fusion with parent op of type {0} at loc {1} is sub-optimal.", dpuOp->getName(), dpuOp->getLoc());
        return false;
    }

    // There might be a way to set proper FP32 clamp values when bypassing conversion to FP16 in FpPPE
    // Ticket to investigate: E#150685
    if (auto postOpIf = mlir::dyn_cast<IE::LayerWithPostOpInterface>(dpuOp)) {
        if (mlir::isa_and_nonnull<IE::ClampAttr>(postOpIf.getPostOp())) {
            log.trace("Parent op of type {0} at loc {1} has Clamp post op.", dpuOp->getName(), dpuOp->getLoc());
            return false;
        }
    }

    return true;
}
