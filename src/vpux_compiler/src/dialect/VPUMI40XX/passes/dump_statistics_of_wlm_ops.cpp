//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_DUMPSTATISTICSOFWLMOPS
#define GEN_PASS_DEF_DUMPSTATISTICSOFWLMOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class DumpStatisticsOfWlmOpsPass : public VPUMI40XX::impl::DumpStatisticsOfWlmOpsBase<DumpStatisticsOfWlmOpsPass> {
public:
    explicit DumpStatisticsOfWlmOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

bool isBarrierProgrammingDMA(VPUMI40XX::NNDMAOp dmaOp) {
    auto declBuff = mlir::dyn_cast<VPURT::DeclareBufferOp>(dmaOp.getOutputBuffs()[0].getDefiningOp());
    if (!declBuff) {
        return false;
    }

    return declBuff.getByteOffset() == VPUMI40XX::FIFO_BARRIERS_NCE_FILL_BARRIER_FIFO_ADR;
}

bool isSHVEnqueueDMA(VPUMI40XX::NNDMAOp dmaOp) {
    auto declBuff = mlir::dyn_cast<VPURT::DeclareBufferOp>(dmaOp.getOutputBuffs()[0].getDefiningOp());
    if (!declBuff) {
        return false;
    }
    const auto offset = declBuff.getByteOffset() - VPUMI40XX::NNCMX_SHV_CMX_CTRL_BASE;
    return llvm::is_contained(VPUMI40XX::SHV_FIFO_OFFSETS, offset);
}

bool isDPUEnqueueDMA(VPUMI40XX::NNDMAOp dmaOp) {
    auto declBuff = mlir::dyn_cast<VPURT::DeclareBufferOp>(dmaOp.getOutputBuffs()[0].getDefiningOp());
    if (!declBuff) {
        return false;
    }
    const auto offset = declBuff.getByteOffset() - VPUMI40XX::NNCMX_DPU_CMX_CTRL_BASE;
    return llvm::is_contained(VPUMI40XX::DPU_FIFO_OFFSETS, offset);
}

void DumpStatisticsOfWlmOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    size_t barrierProgrammingDMACounter = 0;
    size_t enqueueDMACounter = 0;
    llvm::DenseMap<VPURegMapped::TaskType, size_t> enqueueDMACountPerType;

    size_t fetchOpsCounter = 0;
    llvm::DenseMap<VPURegMapped::TaskType, size_t> fetchOpsCountPerType;

    size_t enqueueOpsCounter = 0;
    llvm::DenseMap<VPURegMapped::TaskType, size_t> enqueueOpsCountPerType;

    netFunc->walk([&](mlir::Operation* op) {
        if (auto dmaOp = mlir::dyn_cast<VPUMI40XX::NNDMAOp>(op)) {
            if (auto viewTaskRange =
                        mlir::dyn_cast_or_null<VPURegMapped::ViewTaskRangeOp>(dmaOp.getInput().getDefiningOp())) {
                auto fetchedTaskOp =
                        mlir::dyn_cast<VPURegMapped::TaskOpInterface>(viewTaskRange.getFirst().getDefiningOp());
                VPUX_THROW_WHEN(fetchedTaskOp == nullptr, "Unknow operation fetched by dma - {0}", dmaOp);

                fetchOpsCounter++;
                fetchOpsCountPerType[fetchedTaskOp.getTaskType()]++;
            } else if (isBarrierProgrammingDMA(dmaOp)) {
                barrierProgrammingDMACounter++;
            } else if (isSHVEnqueueDMA(dmaOp)) {
                enqueueDMACounter++;
                enqueueDMACountPerType[VPURegMapped::TaskType::ActKernelInvocation]++;
            } else if (isDPUEnqueueDMA(dmaOp)) {
                enqueueDMACounter++;
                enqueueDMACountPerType[VPURegMapped::TaskType::DPUVariant]++;
            }
        } else if (auto enqueueOp = mlir::dyn_cast<VPURegMapped::EnqueueOp>(op)) {
            enqueueOpsCounter++;
            enqueueOpsCountPerType[enqueueOp.getTaskType()]++;
        }
    });

    _log.info("Fetch DMA count - {0}", fetchOpsCounter);
    for (auto& [taskType, count] : fetchOpsCountPerType) {
        _log.nest().info("{0} - {1}", taskType, count);
    }

    _log.info("WorkItem count - {0}", enqueueOpsCounter);
    for (auto& [taskType, count] : enqueueOpsCountPerType) {
        _log.nest().info("{0} - {1}", taskType, count);
    }

    if (enqueueDMACounter > 0) {
        _log.info("WLM DMA count - {0}", enqueueDMACounter);
        for (auto& [taskType, count] : enqueueDMACountPerType) {
            _log.nest().info("{0} - {1}", taskType, count);
        }
    }

    if (barrierProgrammingDMACounter > 0) {
        _log.info("Barrier Programming DMA count - {0}", barrierProgrammingDMACounter);
    }
}

}  // namespace

//
// createDumpStatisticsOfWlmOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createDumpStatisticsOfWlmOpsPass(Logger log) {
    return std::make_unique<DumpStatisticsOfWlmOpsPass>(log);
}
