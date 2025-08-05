//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool vpux::VPU::isNCEReduceSupported(mlir::Operation* op, LogCb logCb) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<IE::ReduceMeanOp, IE::ReduceSumOp>([&](auto reduceOp) {
                auto axes = IE::extractAxes(reduceOp->getLoc(), reduceOp);
                if (axes.size() != 1 || axes.front() != Dims4D::Act::C.ind()) {
                    logCb(formatv("Axes attribute must be a scalar containing channel dimension index {0}, but got {1}",
                                  Dims4D::Act::C.ind(), axes.front()));
                    return false;
                }
                return true;
            })
            .Default([&](mlir::Operation*) {
                logCb(formatv("Unknown operation: {0}", op->getName()));
                return false;
            });
}

bool vpux::VPU::isReduceOpSupportedOnNCE(mlir::Operation* op) {
    return VPU::getConstraint(op, REDUCE_SUPPORTED);
}

VPUIP::NCETaskType vpux::VPU::configureNCEReduceTaskType(VPU::NCEReduceOp origOp) {
    switch (origOp.getOpType()) {
    case VPU::ReduceType::MEAN:
        return VPUIP::NCETaskType::REDUCEMEAN;
    case VPU::ReduceType::SUM:
        return VPUIP::NCETaskType::REDUCESUM;
    default:
        VPUX_THROW("Unknown nceTaskType {0} ", stringifyReduceType(origOp.getOpType()));
    }
}
