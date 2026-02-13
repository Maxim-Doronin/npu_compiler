//
// Copyright (C) 2023-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_vpu_nce_ops_interface.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace vpux {
#define GEN_PASS_DECL_ONESHOTBUFFERIZEVPU2VPUIP
#define GEN_PASS_DEF_ONESHOTBUFFERIZEVPU2VPUIP
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

void removeBufferizationAttributes(mlir::Operation* op) {
    // removed "__inplace_operands_attr__" on each in-place analyzed operations
    op->walk([&](mlir::Operation* op) {
        if (op->hasAttr("__inplace_operands_attr__")) {
            op->removeAttr("__inplace_operands_attr__");
        }
    });
}

//
// OneshotBufferizeVPU2VPUIPPass
//

class OneShotBufferizeVPU2VPUIPPass final : public impl::OneShotBufferizeVPU2VPUIPBase<OneShotBufferizeVPU2VPUIPPass> {
private:
    void safeRunOnModule() final;
};

void OneShotBufferizeVPU2VPUIPPass::safeRunOnModule() {
    mlir::bufferization::OneShotBufferizationOptions options = vpux::getOneShotBufferizationOptions();
    mlir::ModuleOp moduleOp = getOperation();
    mlir::bufferization::BufferizationState state;
    if (mlir::failed(mlir::bufferization::bufferizeOp(moduleOp, options, state, /*statistics=*/nullptr))) {
        signalPassFailure();
        return;
    }

    removeBufferizationAttributes(moduleOp);
}

}  // namespace

//
// createOneShotBufferizeVPU2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createOneShotBufferizeVPU2VPUIPPass() {
    return std::make_unique<OneShotBufferizeVPU2VPUIPPass>();
}
