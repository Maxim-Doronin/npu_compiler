//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPURT {
#define GEN_PASS_DECL_SATISFYONEWAITBARRIERPERTASK
#define GEN_PASS_DEF_SATISFYONEWAITBARRIERPERTASK
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class SatisfyOneWaitBarrierPerTaskPass final :
        public VPURT::impl::SatisfyOneWaitBarrierPerTaskBase<SatisfyOneWaitBarrierPerTaskPass> {
public:
    explicit SatisfyOneWaitBarrierPerTaskPass(const bool unevenVariantSplitFlag,
                                              std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _unevenVariantSplitFlag(unevenVariantSplitFlag), _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool _mergeWaitBarriersIteratively = true;
    bool _considerTaskExecutorType = true;
    bool _unevenVariantSplitFlag;
    std::optional<WorkloadManagementMode> _workloadManagementMode;
};

void SatisfyOneWaitBarrierPerTaskPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto wlmEnabled = config::getWorkloadManagementStatus(module) == WorkloadManagementStatus::ENABLED;
    if (!wlmEnabled) {  // enforcing single wait barrier per task is not required for non-WLM
        _log.info("SatisfyOneWaitBarrierPerTaskPass skipped as WLM is not enabled");
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    if (_unevenVariantSplitFlag) {
        barrierInfo.enableUnevenVariantSplit();
    }

    const auto maxAvailableSlots = maxVariantCount.hasValue() ? checked_cast<size_t>(maxVariantCount.getValue())
                                                              : VPUIP::getBarrierMaxVariantCount(func);
    const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(func);
    _log.trace("There are {0} slots for each barrier",
               maxSlotsSum < maxAvailableSlots ? maxSlotsSum : maxAvailableSlots);

    const auto availableSlots = vpux::VPUIP::getAvailableSlots(maxSlotsSum, maxAvailableSlots);
    auto mergeBarriersIteratively = mergeWaitBarriersIteratively.hasValue()
                                            ? checked_cast<bool>(mergeWaitBarriersIteratively.getValue())
                                            : _mergeWaitBarriersIteratively;

    // merge parallel wait barriers
    bool modifiedIR = barrierInfo.ensureTasksDrivenBySingleBarrier(availableSlots, mergeBarriersIteratively,
                                                                   _considerTaskExecutorType);

    if (!modifiedIR) {
        // IR was not modified
        barrierInfo.clearAttributes();
        return;
    }

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
    if (!_workloadManagementMode.has_value() ||
        (_workloadManagementMode.value() < WorkloadManagementMode::FWLM_V1_PAGES &&
         _workloadManagementMode.value() != WorkloadManagementMode::PWLM_V0_1_PAGES)) {
        VPUX_THROW_UNLESS(VPURT::verifyBarrierSlots(func, _log), "Barrier slot count check failed");
    }
    auto hasOneWaitBarrierPerTask = VPURT::verifyOneWaitBarrierPerTask(func, _log);
    if (mergeBarriersIteratively) {
        VPUX_THROW_UNLESS(hasOneWaitBarrierPerTask, "Encountered task with more than one wait barrier");
    }
}

}  // namespace

//
// createSatisfyOneWaitBarrierPerTaskPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSatisfyOneWaitBarrierPerTaskPass(
        const bool unevenVariantSplitFlag, std::optional<WorkloadManagementMode> workloadManagementMode, Logger log) {
    return std::make_unique<SatisfyOneWaitBarrierPerTaskPass>(unevenVariantSplitFlag, workloadManagementMode, log);
}
