//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_WRAPVPUIPOPSINNCECLUSTERTILING
#define GEN_PASS_DEF_WRAPVPUIPOPSINNCECLUSTERTILING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

// Utility func to compact inner args of SWKernelOps
void compactSwKernelArgs(VPUIP::SwKernelOp swKernelOp) {
    auto& frontBlock = swKernelOp.getBody().front();
    for (auto& bbArg : frontBlock.getArguments()) {
        bbArg.setType(vpux::VPUIP::getCompactBufferType(bbArg.getType()));
    }

    for (auto swKernelRun : swKernelOp.getBody().getOps<VPUIP::SwKernelRun>()) {
        for (auto runArg : swKernelRun.getArgs()) {
            runArg.setType(vpux::VPUIP::getCompactBufferType(runArg.getType()));
        }
    }
}

//
// WrapVPUIPOpsInNCEClusterTilingPass
//

class WrapVPUIPOpsInNCEClusterTilingPass final :
        public VPUIP::impl::WrapVPUIPOpsInNCEClusterTilingBase<WrapVPUIPOpsInNCEClusterTilingPass> {
public:
    explicit WrapVPUIPOpsInNCEClusterTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//
void WrapVPUIPOpsInNCEClusterTilingPass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](mlir::Operation* origOp) {
        if (!mlir::isa<VPUIP::CopyOp, VPUIP::DMATypeOpInterface, VPUIP::NCEClusterTaskOp, VPUIP::SwKernelOp>(origOp)) {
            return mlir::WalkResult::skip();
        }

        if (origOp->getParentOfType<VPUIP::NCEClusterTilingOp>() != nullptr) {
            _log.trace("Op {0} already wrapped into NCEClusterTilingOp", origOp->getName());
            return mlir::WalkResult::skip();
        }

        // Check for any distributed operands
        auto hasDistributedOperand = llvm::any_of(origOp->getOperands().getTypes(), [](mlir::Type operand) {
            if (auto checkDistributed = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(operand)) {
                return checkDistributed.containsDistributedTypes();
            }
            return false;
        });
        // Check for any distributed results
        bool hasDistributedResult = llvm::any_of(origOp->getResults().getTypes(), [](mlir::Type result) {
            if (auto checkDistributed = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(result)) {
                return checkDistributed.containsDistributedTypes();
            }
            return false;
        });
        // If the op doesn't have DistributedTensor I/O, continue
        if (!hasDistributedOperand && !hasDistributedResult) {
            return mlir::WalkResult::skip();
        }

        if (IE::hasDynamicTensors(origOp)) {
            return mlir::WalkResult::skip();
        }

        mlir::OpBuilder nceBuilder(origOp);
        const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location /*loc*/,
                                          mlir::ValueRange newOperands) {
            mlir::IRMapping mapper;
            mapper.map(origOp->getOperands(), newOperands);
            auto* newOp = builder.clone(*origOp, mapper);
            for (auto operandIter : newOperands | indexed) {
                newOp->setOperand(operandIter.index(), newOperands[operandIter.index()]);
            }

            for (auto resultIter : newOp->getResults()) {
                resultIter.setType(vpux::VPUIP::getCompactBufferType(resultIter.getType()));
            }
            // For SwKernelOps we need to compact their inner args and args of SwKernelRun
            if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(newOp)) {
                compactSwKernelArgs(swKernelOp);
            }
        };

        _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

        auto origOpOperands = origOp->getOperands();
        nceBuilder.setInsertionPoint(origOp);
        auto nceClusterOp = nceBuilder.create<vpux::VPUIP::NCEClusterTilingOp>(
                origOp->getLoc(), origOp->getResultTypes(), origOpOperands, bodyBuilder);

        origOp->replaceAllUsesWith(nceClusterOp.getResults());
        origOp->erase();

        return mlir::WalkResult::advance();
    });
}

}  // namespace

//
// createWrapVPUIPOpsInNCEClusterTilingPass
//

std::unique_ptr<mlir::Pass> VPUIP::createWrapVPUIPOpsInNCEClusterTilingPass(Logger log) {
    return std::make_unique<WrapVPUIPOpsInNCEClusterTilingPass>(log);
}
