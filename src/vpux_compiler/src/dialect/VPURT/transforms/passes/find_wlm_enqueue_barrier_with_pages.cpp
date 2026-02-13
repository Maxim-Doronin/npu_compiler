//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_FINDWLMENQUEUEBARRIERWITHPAGES
#define GEN_PASS_DEF_FINDWLMENQUEUEBARRIERWITHPAGES
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class FindWlmEnqueueBarrierWithPagesPass final :
        public VPURT::impl::FindWlmEnqueueBarrierWithPagesBase<FindWlmEnqueueBarrierWithPagesPass> {
public:
    explicit FindWlmEnqueueBarrierWithPagesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FindWlmEnqueueBarrierWithPagesPass::safeRunOnFunc() {
    auto func = getOperation();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    auto& barrierInfo = getAnalysis<BarrierInfo>();

    auto barrierPagesSplitHandler =
            VPURT::BarrierPagesSplitHandler{func, barrierInfo, static_cast<size_t>(numBarriers), _log};
    barrierPagesSplitHandler.initializeForEnqueue(func);

    auto enqueueBarrierData = barrierPagesSplitHandler.getAndLegalizeEnqueueBarrierData();
    if (enqueueBarrierData.empty()) {
        _log.trace("No enqueue data");
        return;
    }

    barrierInfo = barrierPagesSplitHandler.getUpdatedBarrierInfo();
    barrierInfo.updateIR();

    for (size_t taskInd = 0; taskInd < enqueueBarrierData.size(); ++taskInd) {
        const auto& barOpt = enqueueBarrierData[taskInd];
        if (!barOpt.has_value()) {
            _log.trace("Enqueue task {0} at bootstrap", taskInd);
            continue;
        }

        const auto barInd = barOpt.value();
        _log.trace("Set enqueue barrier {0} for task {1}", barInd, taskInd);
        auto taskOp = barrierInfo.getTaskOpAtIndex(taskInd);
        auto enqBar = barrierInfo.getBarrierOpAtIndex(barInd).getBarrier();
        taskOp.getEnqueueBarrierMutable().assign(enqBar);
    }

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createFindWlmEnqueueBarrierWithPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createFindWlmEnqueueBarrierWithPagesPass(Logger log) {
    return std::make_unique<FindWlmEnqueueBarrierWithPagesPass>(log);
}
