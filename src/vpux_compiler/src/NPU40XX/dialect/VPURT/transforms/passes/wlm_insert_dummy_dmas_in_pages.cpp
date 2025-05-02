//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"

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
    auto arch = VPU::getArch(func);

    if (vpux::VPUIP::getWlmStatus(module) != vpux::VPUIP::WlmStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    SmallVector<VPURT::DeclareVirtualBarrierOp> firstBarOpPerPage;

    _log.trace("Scan all barriers and identify first barrier in each page");
    int prevPage = -1;
    func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto pageOpt = barrierOp.getWlmPage();
        VPUX_THROW_UNLESS(pageOpt.has_value(), "Barrier '{0}' doesn't have WLM page attribute", barrierOp->getLoc());
        auto page = pageOpt.value();

        if (prevPage == -1 || ((page == prevPage + 1) && (firstBarOpPerPage.size() == static_cast<size_t>(page)))) {
            firstBarOpPerPage.push_back(barrierOp);
        }

        prevPage = page;
    });

    _log.trace("Check if each page has a DMA of each type");

    // Keeps track of the last encountered page for each DMA type
    DenseMap<VPURT::TaskQueueType, size_t> dmaTaskOpQueueLastPage;

    // Create dummy input and output buffer that will be needed for creating dummy DMAs
    mlir::OpBuilder builder(func);
    auto buffers = func.getOps<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(buffers.empty(), "Cannot find DeclareBufferOp");
    auto firstDeclareBufferOp = *buffers.begin();
    auto inDdrBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::DDR);
    auto inCmxBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::CMX_NN);
    auto outBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);

    // Traverse all DMA operations and check if in any page there is missing DMA of given type
    func->walk([&](VPURT::TaskOp taskOp) {
        auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
        if (taskQueueType.type != VPU::ExecutorKind::DMA_NN) {
            return;
        }

        const auto attr = taskOp->getAttrOfType<mlir::IntegerAttr>(VPURT::wlmPageAttrName);
        VPUX_THROW_UNLESS(attr != nullptr, "Get: attribute '{0}' was not set for '{1}' operation at '{2}'",
                          VPURT::wlmPageAttrName, taskOp->getName(), taskOp->getLoc());
        auto wlmPage = checked_cast<int>(attr.getValue().getZExtValue());

        int prevTaskPage = -1;
        if (dmaTaskOpQueueLastPage.find(taskQueueType) != dmaTaskOpQueueLastPage.end()) {
            // If this is not the first DMA of this type read previous task page
            prevTaskPage = dmaTaskOpQueueLastPage[taskQueueType];
        }

        // Store information on current DMA page
        dmaTaskOpQueueLastPage[taskQueueType] = wlmPage;

        if (wlmPage - prevTaskPage < 2) {
            // If there is no missing page between two DMA operations of the same type, no dummy DMA is inserted
            return;
        }

        _log.trace("Need to insert tasks on queue DMA[{0}][{1}]", getDMAPortFromEncodedId(taskQueueType.id),
                   getDMAChannelTypeAsString(taskQueueType.id, arch));

        for (int pageInd = prevTaskPage + 1; pageInd < wlmPage; pageInd++) {
            _log.nest().trace("Insert dummy DMA on page {0}", pageInd);

            // Get barrier for given page and get all its users
            // This will be needed to identify insertion point for dummy DMA which wil be placed
            // after all producers and before first consumer of this barrier
            auto firstBarrierOp = firstBarOpPerPage[pageInd];
            auto barrier = firstBarrierOp.getBarrier();
            // Get all barrier users what includes producer and consumer tasks
            auto barOpUsersVec = to_small_vector(barrier.getUsers());

            // From barrier users remove all which update this barrier
            barOpUsersVec.erase(llvm::remove_if(barOpUsersVec,
                                                [&](auto op) {
                                                    auto userTaskOp = mlir::cast<VPURT::TaskOp>(op);
                                                    auto updateBars = userTaskOp.getUpdateBarriers();
                                                    return llvm::find(updateBars, barrier) != updateBars.end();
                                                }),
                                barOpUsersVec.end());

            // Sort remaining users - tasks which wait on this barrier
            llvm::sort(barOpUsersVec, [](mlir::Operation* op1, mlir::Operation* op2) {
                return op1->isBeforeInBlock(op2);
            });

            // Use first task that waits on this barrier as an insertion point
            // Put new dummy DMA before this task
            builder.setInsertionPoint(barOpUsersVec[0]);

            auto inBuffer = (getDMAChannelTypeFromEncodedId(taskQueueType.id, arch) == VPUIP::DmaChannelType::DDR)
                                    ? inDdrBuffer
                                    : inCmxBuffer;
            auto dummyDmaOp = VPUIP::createSyncDMA(builder, inBuffer, outBuffer, 0, {barrier}, {},
                                                   "dummy_dma_fifo_completing_for_wlm_page");
            dummyDmaOp->setAttr(VPURT::wlmPageAttrName, getIntAttr(dummyDmaOp.getContext(), pageInd));
        }
    });
}
}  // namespace

//
// createWlmInsertDummyDmasInPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::arch40xx::createWlmInsertDummyDmasInPagesPass(Logger log) {
    return std::make_unique<WlmInsertDummyDmasInPagesPass>(log);
}
