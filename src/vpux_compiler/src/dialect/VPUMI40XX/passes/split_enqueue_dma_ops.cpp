//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/STLExtras.h>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_SPLITENQUEUEDMAOPS
#define GEN_PASS_DEF_SPLITENQUEUEDMAOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

// DMA(enqueue)  = [var1, var2(breakingPoint), var3, var4(breakingPoint), var5]
// we are going to replace it with 3 enqueue DMAs
// DMA(enqueue1) = [var1, var2]
// DMA(enqueue2) = [var3, var4]
// DMA(enqueue3) = [var5]

class SplitEnqueueDmaOpsPass : public VPUMI40XX::impl::SplitEnqueueDmaOpsBase<SplitEnqueueDmaOpsPass> {
public:
    explicit SplitEnqueueDmaOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    VPUMI40XX::NNDMAOp createEnqueueDma(int64_t startTaskIdx, int64_t endTaskIdx, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers, VPUMI40XX::NNDMAOp origEnqDmaOp,
                                        mlir::Value prevDmaVal);
};

// Create new enqueue DMA based on original one but with updated enqueue DMA attribute task index range and barriers.
VPUMI40XX::NNDMAOp SplitEnqueueDmaOpsPass::createEnqueueDma(int64_t startTaskIdx, int64_t endTaskIdx,
                                                            mlir::ValueRange waitBarriers,
                                                            mlir::ValueRange updateBarriers,
                                                            VPUMI40XX::NNDMAOp origEnqDmaOp, mlir::Value prevDmaVal) {
    mlir::OpBuilder builder(origEnqDmaOp);
    auto newOp = builder.clone(*origEnqDmaOp);
    auto ctx = origEnqDmaOp.getContext();

    auto newEnqueueDmaOp = mlir::cast<VPUMI40XX::NNDMAOp>(newOp);
    newEnqueueDmaOp.getPreviousTaskMutable().clear();
    if (prevDmaVal != nullptr) {
        newEnqueueDmaOp.getPreviousTaskMutable().assign(prevDmaVal);
    }

    newEnqueueDmaOp.getWaitBarriersMutable().clear();
    if (!waitBarriers.empty()) {
        newEnqueueDmaOp.getWaitBarriersMutable().assign(waitBarriers);
    }

    newEnqueueDmaOp.getUpdateBarriersMutable().clear();
    if (!updateBarriers.empty()) {
        newEnqueueDmaOp.getUpdateBarriersMutable().assign(updateBarriers);
    }

    auto enqueueDmaAttr = origEnqDmaOp.getEnqueueDmaAttrAttr();

    auto executorKindAttr = enqueueDmaAttr.getTargetExecutorKindAttr();
    auto tileIdxAttr = enqueueDmaAttr.getTileIdx();
    auto listIdxAttr = enqueueDmaAttr.getListIdx();
    auto startTaskIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), startTaskIdx);
    auto endTaskIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), endTaskIdx);

    auto newEnqueueDmaAttr = VPUIP::EnqueueDMAAttr::get(ctx, executorKindAttr, tileIdxAttr, listIdxAttr,
                                                        startTaskIdxAttr, endTaskIdxAttr);
    newEnqueueDmaOp.setEnqueueDmaAttrAttr(newEnqueueDmaAttr);

    return newEnqueueDmaOp;
}

void SplitEnqueueDmaOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto ctx = &(getContext());
    auto mpi = VPUMI40XX::getMPI(netFunc);

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();
    const auto shavesCountPerTile = IE::getAvailableExecutor(parentModule, VPU::ExecutorKind::SHAVE_ACT).getCount();

    auto dmaTile0List0Head = mpi.getListHead(VPURegMapped::TaskType::DMA, 0, 0);
    if (!dmaTile0List0Head) {
        _log.trace("No list where enqueue DMAs are located");
        return;
    }

    auto firstDmaTile0List0Op = dmaTile0List0Head.getDefiningOp<VPUMI40XX::NNDMAOp>();
    auto enqueueDmasPerHwQueue = VPUMI40XX::getEnqueueDmaData(firstDmaTile0List0Op, _log);

    const mlir::DenseSet<std::pair<VPURegMapped::TaskType, uint32_t>> taskTypesWithListCountPerTile = {
            {{VPURegMapped::TaskType::DPUVariant, 1},
             {VPURegMapped::TaskType::ActKernelInvocation, shavesCountPerTile}}};

    // Iterate over DPU/SHV tasks on each tile and list and check if group end is encountered - presence
    // of lastSecondaryTaskInExecutionGroup attribute. If yes and such task is not the last one that is enqueued
    // by enqueue DMA then such enqueue DMA needs to be split - old enqueue DMA is erased and multiple new ones are
    // created. When split is done enqueueDmaAttribute start/end task indexes are updated and barrier dependencies
    // changed so that only the first DMA resulting from split has wait barrier and only last one has update barrier.
    // Example:
    //   Task0, Task1(lastSecondaryTaskInExecutionGroup), Task2, Task3(lastSecondaryTaskInExecutionGroup)
    // Before:
    //   DMA {enqueueDmaAttribute(startTaskIdx=0, endTaskIdx=3)} waitBarrier(BAR0) updateBarrier(BAR1)
    // After:
    //   DMA {enqueueDmaAttribute(startTaskIdx=0, endTaskIdx=1)} waitBarrier(BAR0)
    //   DMA {enqueueDmaAttribute(startTaskIdx=2, endTaskIdx=3)} updateBarrier(BAR1)
    //
    for (uint32_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (const auto& [taskType, listCount] : taskTypesWithListCountPerTile) {
            for (uint32_t listIdx = 0; listIdx < listCount; listIdx++) {
                auto listHead = mpi.getListHead(taskType, tileIdx, listIdx);
                if (!listHead) {
                    continue;
                }

                _log.trace("Check task type {0} on tile {1}, list {2} if enqueue DMA split is needed", taskType,
                           tileIdx, listIdx);
                auto taskOp = mlir::cast<VPURegMapped::TaskOpInterface>(listHead.getDefiningOp());

                auto hwQueue = VPUMI40XX::HwQueueType{taskType, tileIdx, listIdx};

                VPUX_THROW_WHEN(enqueueDmasPerHwQueue.find(hwQueue) == enqueueDmasPerHwQueue.end(),
                                "No Enqueue DMAs available for task type {0} on tile {1}, list {2}", taskType, tileIdx,
                                listIdx);
                // Initial tasks must be enqueued by first enqueue DMA for given HW queue type
                size_t curEnqueueIndex = 0;
                // Get start and end task range enqueued by enqueue DMA to understand what range of tasks
                // are processed by a single enqueue DMA
                auto [enqueueDmaStartIdx, enqueueDmaEndIdx, enqueueDmaOp] =
                        enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex];
                _log.trace("Enqueue DMA task range: {0} - {1}", enqueueDmaStartIdx, enqueueDmaEndIdx);

                // When splitting DMAs and creating new ones code needs to correctly maintain connections
                // to previous DMA
                auto prevDmaVal = enqueueDmaOp.getPreviousTask();

                _log = _log.nest();
                bool breakPointDetected = false;
                // Flag to indicate if when encountering next breakpoint it is the first one for given enqueue DMA.
                // This is used to indicate which newly created enqueue DMA should have wait barriers - only the first
                // one
                bool nextBreakPointIsFirstForGivenEnqueueDma = true;
                // Iterate over tasks in the list and check for lastSecondaryTaskInExecutionGroup
                do {
                    auto taskInd = mlir::cast<VPURegMapped::IndexType>(taskOp.getResult().getType()).getValue();
                    // If current task index is greater than end index of current enqueue DMA it means that
                    // this task is enqueued by next enqueue DMA -> switch to next enqueue DMA
                    if (taskInd > enqueueDmaEndIdx) {
                        _log.trace("Task {0} is after end task {1} of current enqueue DMA. Move to next enqueue DMA op",
                                   taskInd, enqueueDmaEndIdx);
                        if (breakPointDetected) {
                            // If breakpoint was detected before, when switching to using next enqueue DMA first need to
                            // create new enqueue DMA that will handle enqueue of last tasks for this enqueue DMA
                            _log.trace("Create new enqueue DMA with range {0} - {1} to handle enqueue of last tasks in "
                                       "group",
                                       enqueueDmaStartIdx, enqueueDmaEndIdx);
                            auto newEnqueueDmaOp =
                                    createEnqueueDma(enqueueDmaStartIdx, enqueueDmaEndIdx, {},
                                                     enqueueDmaOp.getUpdateBarriers(), enqueueDmaOp, prevDmaVal);

                            enqueueDmaOp.replaceAllUsesWith(newEnqueueDmaOp.getIndex());
                            enqueueDmaOp.erase();
                        }
                        // Move to next enqueue DMA and get start and end indexes
                        curEnqueueIndex++;
                        VPUX_THROW_UNLESS(
                                curEnqueueIndex < enqueueDmasPerHwQueue[hwQueue].size(),
                                "No enqueue DMAs available for task type {0} on tile {1}, list {2} at index {3}",
                                taskType, tileIdx, listIdx, curEnqueueIndex);
                        enqueueDmaStartIdx = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex].startTaskIdx;
                        enqueueDmaEndIdx = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex].endTaskIdx;
                        enqueueDmaOp = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex].enqDmaOp;
                        prevDmaVal = enqueueDmaOp.getPreviousTask();
                        breakPointDetected = false;
                        nextBreakPointIsFirstForGivenEnqueueDma = true;
                        _log.trace("Enqueue DMA task range: {0} - {1}", enqueueDmaStartIdx, enqueueDmaEndIdx);
                    }
                    VPUX_THROW_UNLESS(taskInd >= enqueueDmaStartIdx && taskInd <= enqueueDmaEndIdx,
                                      "Task index {0} is out of range for Enqueue DMA: {1} - {2}", taskInd,
                                      enqueueDmaStartIdx, enqueueDmaEndIdx);

                    // Check if current task is the last one in execution group but also is not the last one
                    // that is enqueued by enqueue DMA. In such case enqueue DMA needs to be split
                    if (taskOp->hasAttr(VPUMI40XX::lastSecondaryTaskInExecutionGroup) && taskInd < enqueueDmaEndIdx) {
                        _log.trace("Task index {0} - breakpoint detected", taskInd);
                        breakPointDetected = true;

                        _log.trace("Create new enqueue DMA with range {0} - {1}", enqueueDmaStartIdx, taskInd);
                        mlir::ValueRange waitBarriers = {};
                        if (nextBreakPointIsFirstForGivenEnqueueDma) {
                            waitBarriers = enqueueDmaOp.getWaitBarriers();
                        }
                        nextBreakPointIsFirstForGivenEnqueueDma = false;

                        auto newEnqueueDmaOp = createEnqueueDma(enqueueDmaStartIdx, taskInd, waitBarriers, {},
                                                                enqueueDmaOp, prevDmaVal);
                        prevDmaVal = newEnqueueDmaOp.getIndex();

                        // If this enqueue DMA is also first one in the list then when replacing it with new one update
                        // its usage in MappedInferenceOp and EnqueueOp responsible for enqueueing this DMA
                        if (firstDmaTile0List0Op == enqueueDmaOp) {
                            firstDmaTile0List0Op.getResult().replaceUsesWithIf(
                                    newEnqueueDmaOp, [](mlir::OpOperand& operand) {
                                        if (mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner())) {
                                            return true;
                                        }

                                        if (auto enqueueOp =
                                                    mlir::dyn_cast<VPURegMapped::EnqueueOp>(operand.getOwner())) {
                                            if (enqueueOp.getStartMutable() == operand) {
                                                return true;
                                            }
                                        }

                                        return false;
                                    });
                            firstDmaTile0List0Op = newEnqueueDmaOp;
                        }

                        // If new enqueue DMA was created then update the enqueueDmaStartIdx to point to the next task
                        // so that next new enqueue DMA will have correct start index
                        enqueueDmaStartIdx = taskInd + 1;
                    }
                    taskOp = taskOp.getNextTask();
                } while (taskOp);

                if (breakPointDetected) {
                    _log.trace("After processing all tasks create new enqueue DMA with range {0} - {1} to handle "
                               "enqueue of last tasks of this type",
                               enqueueDmaStartIdx, enqueueDmaEndIdx);
                    auto newEnqueueDmaOp = createEnqueueDma(enqueueDmaStartIdx, enqueueDmaEndIdx, {},
                                                            enqueueDmaOp.getUpdateBarriers(), enqueueDmaOp, prevDmaVal);
                    enqueueDmaOp.replaceAllUsesWith(newEnqueueDmaOp.getIndex());
                    enqueueDmaOp.erase();
                }

                _log = _log.unnest();
            }
        }
    }

    // After splitting enqueue DMAs their indexes and count needs to be updated
    auto newCount = VPUMI40XX::reindexList(firstDmaTile0List0Op);
    auto dmasCount = parseIntArrayOfArrayAttr<int64_t>(mpi.getDmaCount());
    dmasCount[0][0] = newCount;
    mpi.setDmaCountAttr(getIntArrayOfArray(ctx, dmasCount));
}

}  // namespace

//
// createSplitEnqueueDmaOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createSplitEnqueueDmaOpsPass(Logger log) {
    return std::make_unique<SplitEnqueueDmaOpsPass>(log);
}
