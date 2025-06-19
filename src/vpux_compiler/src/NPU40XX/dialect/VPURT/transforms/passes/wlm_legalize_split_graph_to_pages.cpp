//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

namespace vpux::VPURT::arch40xx {
#define GEN_PASS_DECL_WLMLEGALIZESPLITGRAPHTOPAGES
#define GEN_PASS_DEF_WLMLEGALIZESPLITGRAPHTOPAGES
#include "vpux/compiler/NPU40XX/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT::arch40xx

using namespace vpux;

namespace {

class WlmLegalizeSplitGraphToPagesPass final :
        public VPURT::arch40xx::impl::WlmLegalizeSplitGraphToPagesBase<WlmLegalizeSplitGraphToPagesPass> {
public:
    explicit WlmLegalizeSplitGraphToPagesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WlmLegalizeSplitGraphToPagesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    if (vpux::VPUIP::getWlmStatus(module) != vpux::VPUIP::WlmStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForLegalization();

    if (barrierPagesSplitHandler.isSplitToPagesValid()) {
        barrierInfo.clearAttributes();
        return;
    }

    _log.trace("Schedule needs to be legalized for barrier page split");
    barrierPagesSplitHandler.legalizeScheduleForBarrierPagesSplit();

    // Verify if legalization was successful and split is valid afterwards
    VPUX_THROW_UNLESS(barrierPagesSplitHandler.isSplitToPagesValid(), "Split to pages is not valid");

    barrierPagesSplitHandler.updateIR();
    barrierInfo = vpux::BarrierInfo{func};
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);

    VPUX_THROW_UNLESS(VPURT::verifyBarrierSlots(func, _log), "Barrier slot count check failed");
}
}  // namespace

//
// createWlmLegalizeSplitGraphToPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::arch40xx::createWlmLegalizeSplitGraphToPagesPass(Logger log) {
    return std::make_unique<WlmLegalizeSplitGraphToPagesPass>(log);
}
