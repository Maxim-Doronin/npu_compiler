//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/async_dialect_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"

#include <mlir/Dialect/Async/IR/Async.h>

namespace vpux::VPUIP {

config::ExecutorKind getExecutorType(mlir::async::ExecuteOp execOp) {
    if (execOp->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
        return VPUIP::VPUIPDialect::getExecutorKind(execOp);
    }
    // treat all other executors as DPU
    return config::ExecutorKind::DPU;
}

VPUIP::DMATypeOpInterface getDmaTypeOp(mlir::async::ExecuteOp execOp) {
    auto* bodyBlock = execOp.getBody();

    for (auto& op : bodyBlock->getOperations()) {
        if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
            return dmaOp;
        }
    }

    return nullptr;
}
}  // namespace vpux::VPUIP
