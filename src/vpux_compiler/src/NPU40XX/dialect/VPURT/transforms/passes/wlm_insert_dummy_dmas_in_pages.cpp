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
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT::arch40xx {
#define GEN_PASS_DECL_WLMINSERTDUMMYDMASINPAGES
#define GEN_PASS_DEF_WLMINSERTDUMMYDMASINPAGES
#include "vpux/compiler/NPU40XX/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT::arch40xx

using namespace vpux;

namespace {

class WlmInsertDummyDmasInPagesPass final :
        public VPURT::arch40xx::impl::WlmInsertDummyDmasInPagesBase<WlmInsertDummyDmasInPagesPass> {
public:
    explicit WlmInsertDummyDmasInPagesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WlmInsertDummyDmasInPagesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto arch = config::getArch(func);

    if (VPU::getWorkloadManagementStatus(module) != VPU::WorkloadManagementStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForLegalization();
    auto dummyDmaInsertionDataVec = barrierPagesSplitHandler.getAndLegalizeDummyDmaInsertionData();

    if (dummyDmaInsertionDataVec.empty()) {
        return;
    }

    _log.trace("Insert {0} dummy DMAs in pages", dummyDmaInsertionDataVec.size());

    // Retrieving dummy DMA data may introduce new dependencies which need to be updated in IR
    // Recreate barrier info to have latest status of the IR
    barrierPagesSplitHandler.updateIR();
    barrierInfo = vpux::BarrierInfo{func};

    // Create dummy input and output buffer that will be needed for creating dummy DMAs
    mlir::OpBuilder builder(func);
    auto buffers = func.getOps<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(buffers.empty(), "Cannot find DeclareBufferOp");
    auto firstDeclareBufferOp = *buffers.begin();
    auto inDdrBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::DDR);
    auto inCmxBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::CMX_NN);
    auto outBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);

    for (auto dummyDmaInsertionData : dummyDmaInsertionDataVec) {
        auto pageInd = dummyDmaInsertionData.pageInd;
        auto queueType = dummyDmaInsertionData.queueType;
        auto insertionPointOp = barrierInfo.getTaskOpAtIndex(dummyDmaInsertionData.insertAfter);
        auto port = getDMAPortFromEncodedId(queueType.id);

        _log.trace("Insert new DMA[{0}][{1}] in page {2}", port, getDMAChannelTypeAsString(queueType.id, arch),
                   pageInd);

        builder.setInsertionPointAfter(insertionPointOp);

        auto inBuffer = (getDMAChannelTypeFromEncodedId(queueType.id, arch) == VPUIP::DmaChannelType::DDR)
                                ? inDdrBuffer
                                : inCmxBuffer;
        auto dummyDmaOp = VPUIP::createSyncDMA(builder, inBuffer, outBuffer, port, {}, {},
                                               "dummy_dma_fifo_completing_for_wlm_page");
        dummyDmaOp.setWlmPage(pageInd);

        auto dummyDmaOpTaskInd = barrierInfo.addNewTaskOp(dummyDmaOp);

        llvm::for_each(dummyDmaInsertionData.waitBars, [&](auto waitBar) {
            _log.nest().trace("wait bar {0}", waitBar);
            barrierInfo.addConsumer(waitBar, dummyDmaOpTaskInd);
        });

        llvm::for_each(dummyDmaInsertionData.updateBars, [&](auto updateBar) {
            _log.nest().trace("update bar {0}", updateBar);
            barrierInfo.addProducer(updateBar, dummyDmaOpTaskInd);
        });
    }
    barrierInfo.updateIR();
    barrierInfo = vpux::BarrierInfo{func};

    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createWlmInsertDummyDmasInPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::arch40xx::createWlmInsertDummyDmasInPagesPass(Logger log) {
    return std::make_unique<WlmInsertDummyDmasInPagesPass>(log);
}
