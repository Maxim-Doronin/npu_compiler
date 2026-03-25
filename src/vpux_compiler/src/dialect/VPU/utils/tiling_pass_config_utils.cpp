//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_pass_config_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool VPU::hasDynamicDimAlignment(mlir::Operation* operation) {
    mlir::func::FuncOp func = config::getOwningFuncOp(operation);
    return func->hasAttr(DYNAMIC_DIM_ALIGNMENT);
}

void VPU::setDynamicDimAlignment(mlir::Operation* operation) {
    mlir::func::FuncOp func = config::getOwningFuncOp(operation);
    // UnitAttr is used to mark the presence of dynamic dimension alignment
    func->setAttr(DYNAMIC_DIM_ALIGNMENT, mlir::UnitAttr::get(func.getContext()));
}

void VPU::removeDynamicDimAlignment(mlir::Operation* operation) {
    mlir::func::FuncOp func = config::getOwningFuncOp(operation);
    if (func->hasAttr(DYNAMIC_DIM_ALIGNMENT)) {
        func->removeAttr(DYNAMIC_DIM_ALIGNMENT);
    }
}
