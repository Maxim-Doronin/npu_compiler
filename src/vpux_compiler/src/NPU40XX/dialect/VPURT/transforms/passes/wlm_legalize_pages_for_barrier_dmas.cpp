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
#define GEN_PASS_DECL_WLMLEGALIZEPAGESFORBARRIERDMAS
#define GEN_PASS_DEF_WLMLEGALIZEPAGESFORBARRIERDMAS
#include "vpux/compiler/NPU40XX/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT::arch40xx

using namespace vpux;

namespace {

class WlmLegalizePagesForBarrierDmasPass final :
        public VPURT::arch40xx::impl::WlmLegalizePagesForBarrierDmasBase<WlmLegalizePagesForBarrierDmasPass> {
public:
    explicit WlmLegalizePagesForBarrierDmasPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// DMA programs 4 instances for each pid in range
VPUIP::PhysicalBarrierRangeAttr getPidRangeForDMAProgBarrier(mlir::OpBuilder& builder, int64_t nPhysBarrs,
                                                             int64_t barPDmaPage) {
    int64_t pidStart = 0;
    int64_t pidEnd = nPhysBarrs - 1;

    if (barPDmaPage % 2 == 1) {
        // Odd pages → First half of barriers
        pidEnd = (nPhysBarrs / 2) - 1;
    } else {
        // Even pages → Second half of barriers
        pidStart = (nPhysBarrs / 2);
    }

    auto pidStartAttr = mlir::IntegerAttr::get(getInt64Type(builder.getContext()), pidStart);
    auto pidEndAttr = mlir::IntegerAttr::get(getInt64Type(builder.getContext()), pidEnd);
    return VPUIP::PhysicalBarrierRangeAttr::get(builder.getContext(), pidStartAttr, pidEndAttr);
}

void WlmLegalizePagesForBarrierDmasPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nPhysBarrs = VPUIP::getNumAvailableBarriers(func);

    if (vpux::VPUIP::getWlmStatus(module) != vpux::VPUIP::WlmStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }
    const auto numBarriers = numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : nPhysBarrs;

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(barrierInfo, numBarriers, _log);
    if (barrierFifoDepthOpt.hasValue()) {
        // In case pass option has barrier FIFO depth set, use it. Otherwise module
        // will use default FIFO depth (4)
        barrierPagesSplitHandler.reconfigureBarrierFifoDepth(barrierFifoDepthOpt.getValue());
    }
    barrierPagesSplitHandler.initializeForLegalization();
    barrierPagesSplitHandler.legalizeForDmaProgrammingBarriers();
    // Legalization for barrier DMAs might introduce new dependencies which need to be updated in IR
    // Recreate barrier info to have latest status of the IR
    barrierPagesSplitHandler.updateIR();
    barrierInfo = vpux::BarrierInfo{func};
    auto barProgDmas = barrierPagesSplitHandler.getDmaProgrammingBarrierPositions();

    // Create dummy input and output buffer
    mlir::OpBuilder builder(func);
    auto buffers = func.getOps<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(buffers.empty(), "Cannot find DeclareBufferOp");
    auto firstDeclareBufferOp = *buffers.begin();
    auto inBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);
    auto outBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);

    for (const auto& [pageInd, barProgDma] : barProgDmas | indexed) {
        if (!barProgDma.valid) {
            continue;
        }

        auto waitBars = barProgDma.waitBars;
        auto updateBars = barProgDma.updateBars;

        // Track insertPoint as taskIndex so we can get the Op from it
        // The page assigned to this Op is also assigned to the SyncDMAOp
        auto insertPoint = barProgDma.insertAfter;
        VPUX_THROW_UNLESS(waitBars.size() >= 1, "Invalid wait bars count");

        _log.trace("Barrier programming DMA for page {0}, insert after {1}:", pageInd, barProgDma.insertAfter);
        // Create DMAs for each wait bar to satisfy 1-wait bar condition
        // Last DMA would be the barrier programming DMA
        auto dmaCount = waitBars.size();
        for (size_t i = 0; i < dmaCount - 1; i++) {
            auto waitBar = waitBars[i];

            // Create DMA for wait bar
            _log.nest().trace("Create sync DMA with wait bar {0}", waitBar);

            auto insertPointOp = barrierInfo.getTaskOpAtIndex(insertPoint);
            builder.setInsertionPointAfter(insertPointOp);

            auto syncTaskOp =
                    VPUIP::createSyncDMA(builder, inBuffer, outBuffer, 0, {}, {}, "pre_bar_reprogram_sync_dma");
            syncTaskOp.setWlmPageAttr(insertPointOp.getWlmPageAttr());

            auto syncTaskInd = barrierInfo.addNewTaskOp(syncTaskOp);
            barrierInfo.addConsumer(waitBar, syncTaskInd);

            insertPoint = syncTaskInd;
            syncTaskOp.setWlmPage(pageInd);
        }

        // Create placeholder DMA for barrier reprogramming
        _log.nest().trace("Create barrier programming DMA with wait bar {0}", waitBars.back());

        auto insertPointOp = barrierInfo.getTaskOpAtIndex(insertPoint);
        builder.setInsertionPointAfter(insertPointOp);
        auto pidAttr = getPidRangeForDMAProgBarrier(builder, numBarriers, pageInd);
        auto barProgDMATaskOp = VPUIP::createBarProgDMA(builder, inBuffer, outBuffer, 0, {}, {}, pidAttr);
        barProgDMATaskOp.setWlmPage(pageInd);

        auto barProgDMATaskInd = barrierInfo.addNewTaskOp(barProgDMATaskOp);
        barrierInfo.addConsumer(waitBars.back(), barProgDMATaskInd);
        llvm::for_each(updateBars, [&](auto updateBar) {
            _log.nest(2).trace("update bar {0}", updateBar);
            barrierInfo.addProducer(updateBar, barProgDMATaskInd);
        });
    }

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);

    VPUX_THROW_UNLESS(VPURT::verifyBarrierSlots(func, _log), "Barrier slot count check failed");
}
}  // namespace

//
// createWlmLegalizePagesForBarrierDmasPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::arch40xx::createWlmLegalizePagesForBarrierDmasPass(Logger log) {
    return std::make_unique<WlmLegalizePagesForBarrierDmasPass>(log);
}
