//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

namespace vpux::VPURT::arch40xx {
#define GEN_PASS_DECL_WLMSPLITGRAPHTOPAGES
#define GEN_PASS_DEF_WLMSPLITGRAPHTOPAGES
#include "vpux/compiler/NPU40XX/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT::arch40xx

using namespace vpux;

namespace {

class WlmSplitGraphToPagesPass final :
        public VPURT::arch40xx::impl::WlmSplitGraphToPagesBase<WlmSplitGraphToPagesPass> {
public:
    explicit WlmSplitGraphToPagesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WlmSplitGraphToPagesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    if (vpux::VPUIP::getWlmStatus(module) != vpux::VPUIP::WlmStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    if (!VPURT::verifyOneWaitBarrierPerTask(func, _log)) {
        _log.warning("WLM cannot be enabled as not all tasks have 1 wait barrier");
        vpux::VPUIP::setWlmStatus(module, vpux::VPUIP::WlmStatus::FAILED);
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForAssignment(func);
    barrierPagesSplitHandler.assignPagesToBarriersInIr();
    barrierPagesSplitHandler.assignPagesToTasksInIr();
    barrierInfo.clearAttributes();
}
}  // namespace

//
// createWlmSplitGraphToPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::arch40xx::createWlmSplitGraphToPagesPass(Logger log) {
    return std::make_unique<WlmSplitGraphToPagesPass>(log);
}
