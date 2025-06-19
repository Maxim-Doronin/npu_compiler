
//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/async_dialect_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

using namespace vpux;

mlir::Type vpux::getAsyncValueType(mlir::Value value) {
    auto type = value.getType();
    if (const auto asyncType = mlir::dyn_cast<mlir::async::ValueType>(type)) {
        type = asyncType.getValueType();
    }
    return type;
}

VPUIP::DMATypeOpInterface vpux::getDmaTypeOp(mlir::async::ExecuteOp execOp) {
    auto* bodyBlock = execOp.getBody();

    for (auto& op : bodyBlock->getOperations()) {
        if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
            return dmaOp;
        }
    }

    return nullptr;
}

VPU::ExecutorKind vpux::getExecutorType(mlir::async::ExecuteOp execOp) {
    if (execOp->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
        return VPUIP::VPUIPDialect::getExecutorKind(execOp);
    }
    // treat all other executors as DPU
    return VPU::ExecutorKind::DPU;
}
