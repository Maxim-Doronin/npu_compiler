//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/workload_management_status_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT::arch40xx {
#define GEN_PASS_DECL_FINDWLMENQUEUEDMASBARRIER
#define GEN_PASS_DEF_FINDWLMENQUEUEDMASBARRIER
#include "vpux/compiler/NPU40XX/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT::arch40xx

using namespace vpux;

namespace {

class FindWlmEnqueueDmasBarrierPass final :
        public VPURT::arch40xx::impl::FindWlmEnqueueDmasBarrierBase<FindWlmEnqueueDmasBarrierPass> {
public:
    explicit FindWlmEnqueueDmasBarrierPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FindWlmEnqueueDmasBarrierPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (VPU::getWorkloadManagementStatus(module) != VPU::WorkloadManagementStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    auto& barrierInfo = getAnalysis<BarrierInfo>();

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForEnqueue(func);

    mlir::DenseSet<vpux::VPU::ExecutorKind> executorEnqAtBootstrap{vpux::VPU::ExecutorKind::DMA_NN};

    auto enqueueBarVec = barrierPagesSplitHandler.prepareEnqueueDmaBarForFullWlm(executorEnqAtBootstrap);

    VPUX_THROW_WHEN(enqueueBarVec.empty(), "No enqueue DMA barrier data created");

    for (size_t taskInd = 0; taskInd < enqueueBarVec.size(); ++taskInd) {
        const auto& enqBar = enqueueBarVec[taskInd];
        _log.trace("Enqueue task {0} at barrier {1}", taskInd,
                   (!enqBar.has_value() ? "BOOTSTRAP" : std::to_string(enqBar.value())));

        if (enqBar.has_value()) {
            auto taskOp = barrierInfo.getTaskOpAtIndex(taskInd);
            auto barrier = barrierInfo.getBarrierOpAtIndex(enqBar.value()).getBarrier();
            taskOp.getEnqueueBarrierMutable().assign(barrier);
        }
    }

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createFindWlmEnqueueDmasBarrierPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::arch40xx::createFindWlmEnqueueDmasBarrierPass(Logger log) {
    return std::make_unique<FindWlmEnqueueDmasBarrierPass>(log);
}
