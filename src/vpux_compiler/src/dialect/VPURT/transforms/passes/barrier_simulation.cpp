//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_BARRIERSIMULATION
#define GEN_PASS_DEF_BARRIERSIMULATION
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

void setBarrierAttributes(mlir::MLIRContext* ctx, BarrierInfo& barrierInfo, const ExecutionGroupList& fifoExecGroups,
                          size_t blockIdx, std::pair<SmallVector<llvm::BitVector>, size_t>& taskControlMapAndOffset,
                          std::optional<size_t>& blockIdxOfTaskControlMap,
                          SmallVector<size_t> barConsumptionReadyOrder = {}) {
    auto hasPath = [&](size_t taskA, size_t taskB) {
        return barrierInfo.isDepFromTaskAToTaskB(taskA, taskB, taskControlMapAndOffset, blockIdxOfTaskControlMap);
    };

    auto hasNoConsumersDependentOnGrandChildGrp = [&](size_t grpIdx, size_t barIdx) {
        auto firstTaskInGrandChildGroupInBlock =
                barrierInfo.getFirstTaskInGroupFromBlock(grpIdx + 2, blockIdx, fifoExecGroups);

        if (!firstTaskInGrandChildGroupInBlock.has_value()) {
            // First task in grand child group is in different block or grand child group does not exist.
            // In either case the update barrier from the end of the current group is legal because it does not have
            // consumers dependant on grand child group.
            return true;
        }

        auto barrierConsumers = barrierInfo.getBarrierConsumers(barIdx);
        for (const auto& consumer : barrierConsumers) {
            if (hasPath(firstTaskInGrandChildGroupInBlock.value(), consumer)) {
                return false;
            }
        }

        return true;
    };

    auto getLegalBarrierIdx = [&](size_t grpIdx, auto& endGrpUpdateBarriers) {
        // Find index of an update barrier that does not have consumers dependant on grand child group
        for (auto endGrpUpdateBarrier : endGrpUpdateBarriers) {
            auto barrierConsumers = barrierInfo.getBarrierConsumers(endGrpUpdateBarrier);

            auto firstTaskInGrandChildGroupInBlock =
                    barrierInfo.getFirstTaskInGroupFromBlock(grpIdx + 2, blockIdx, fifoExecGroups);

            if (!firstTaskInGrandChildGroupInBlock.has_value()) {
                // First task in grand child group is in different block or grand child group does not exist.
                // In either case the update barrier from the end of the current group is legal because it does not have
                // consumers dependant on grand child group.
                return endGrpUpdateBarrier;
            }

            bool hasConsumersDependentOnGrandChildGrp = false;
            for (const auto& consumer : barrierConsumers) {
                if (hasPath(firstTaskInGrandChildGroupInBlock.value(), consumer)) {
                    hasConsumersDependentOnGrandChildGrp = true;
                    break;
                }
            }

            if (!hasConsumersDependentOnGrandChildGrp) {
                return endGrpUpdateBarrier;
            }
        }
        VPUX_THROW("No legal barrier found");
    };

    // Go over execution groups and calculate and assign clean_after descriptor fields
    auto numberOfGroups = static_cast<size_t>(fifoExecGroups.size());
    size_t cleanAfter = std::numeric_limits<size_t>::max();  // dummy value
    for (const auto& [grpIdx, execGroup] : enumerate(fifoExecGroups)) {
        auto lastTaskInGroup = *execGroup.rbegin();
        if (barrierInfo.getControlGraphBlockIndex(lastTaskInGroup) != blockIdx) {
            continue;
        }

        if (grpIdx + 2 >= numberOfGroups) {
            // for tasks from the last or last but one execution group, return the final barrier, as the last but one
            // execution group may not have an update barrier.
            cleanAfter = barrierInfo.getNumOfBarrierOps() - 1;
        } else {
            auto endGrpUpdBars = barrierInfo.getUpdateBarriers(lastTaskInGroup);
            VPUX_THROW_WHEN(endGrpUpdBars.empty(),
                            "Last task ({0}) in execution group {1}/{2} does not have update barrier", lastTaskInGroup,
                            grpIdx, numberOfGroups);

            if (endGrpUpdBars.size() == 1) {
                cleanAfter = *endGrpUpdBars.begin();
            } else {
                cleanAfter = getLegalBarrierIdx(grpIdx, endGrpUpdBars);
            }
        }
        auto legalBarrierIdx = cleanAfter;

        auto getEarliestCleanAfterBarrier = [&](auto taskIdx, auto grpIdx, auto legalBarrierIdx) {
            auto taskUpdateBars = barrierInfo.getUpdateBarriers(taskIdx);
            size_t earliestCleanAfter = legalBarrierIdx;
            for (const auto& updBar : taskUpdateBars) {
                if (barConsumptionReadyOrder[updBar] < barConsumptionReadyOrder[earliestCleanAfter] &&
                    hasNoConsumersDependentOnGrandChildGrp(grpIdx, updBar)) {
                    earliestCleanAfter = updBar;
                }
            }

            return earliestCleanAfter;
        };

        // set attributes for all tasks in the group
        for (const auto& taskIdx : execGroup) {
            cleanAfter = getEarliestCleanAfterBarrier(taskIdx, grpIdx, legalBarrierIdx);

            auto op = barrierInfo.getTaskOpAtIndex(taskIdx);
            auto newCleanAfterAttr =
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned), cleanAfter);
            op.setCleanAfterAttr(newCleanAfterAttr);
        }
    }
}

//
// BarrierSimulationPass
//

class BarrierSimulationPass final : public VPURT::impl::BarrierSimulationBase<BarrierSimulationPass> {
public:
    explicit BarrierSimulationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void verifyCleanAfterAttribute(BarrierInfo& barrierInfo, config::ExecutorKind execType) {
        for (size_t taskIdx = 0; taskIdx < barrierInfo.getNumOfTasks(); ++taskIdx) {
            auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
            if (taskOp.getExecutorKind() != execType) {
                continue;
            }
            auto cleanAfterAttr = taskOp.getCleanAfterAttr();
            VPUX_THROW_UNLESS(cleanAfterAttr != nullptr,
                              "{0} task at index {1} does not have clean_after attribute: {2}", execType, taskIdx,
                              taskOp);
            auto cleanAfterVal =
                    static_cast<size_t>(mlir::cast<mlir::IntegerAttr>(cleanAfterAttr).getValue().getZExtValue());
            VPUX_THROW_UNLESS(cleanAfterVal < barrierInfo.getNumOfBarrierOps(),
                              "Task at index {0} has invalid clean_after attribute value {1}", taskIdx, cleanAfterVal);
        }
    }

private:
    void safeRunOnFunc() final;
};

void BarrierSimulationPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto ctx = funcOp.getContext();
    // NPU37XX has different mechanism for assuring safe task descriptors fetch (i.e. with NPU37XX, DPU tasks always
    // utilize nVariants barrier slots) and clean_after fields don't need to be set here. They will be set in
    // BarrierComputation pass.
    auto configureDescriptorFieldsForTaskFetch = !config::isArchVPUX3XXX(config::getArch(funcOp));

    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    auto wlmFlag = config::getWorkloadManagementStatus(module) == WorkloadManagementStatus::ENABLED;

    if (wlmFlag) {
        // No need to set clean_after fields as they are needed only in case of nonWLM config
        configureDescriptorFieldsForTaskFetch = false;
    }

    if (configureDescriptorFieldsForTaskFetch) {
        auto barrierInfo = vpux::BarrierInfo{funcOp};
        // For non-WLM mode, task descriptor fields clean_after need to be set so as to ensure descriptors buffer is
        // released for fetching the next set of tasks
        ExecutionGroupAnalysis execGroupAnalysis(funcOp, /* ignoreVariantLimit */ true,
                                                 /* ignoreInvariantLimit */ false);

        auto barConsumptionReadyOrder =
                VPURT::getBarriersOrder(funcOp, barrierInfo, /* orderByConsumptionReady */ true);

        barrierInfo.buildTaskQueueTypeMap();

        for (size_t taskBlockIndex = 0; taskBlockIndex < barrierInfo.getControlGraphBlockCount(); ++taskBlockIndex) {
            // build task control map for current block and all executor kinds
            auto taskControlMapAndOffset =
                    barrierInfo.buildTaskControlMap(taskBlockIndex, /* considerTaskFifoDependency */ true);
            std::optional<size_t> blockIdxOfTaskControlMap = taskBlockIndex;

            _log.trace("Set barrier attributes for DPU in block {0}", taskBlockIndex);
            for (const auto& [queue, fifoExecGroups] : execGroupAnalysis.getDPUExecutionGroups()) {
                setBarrierAttributes(ctx, barrierInfo, fifoExecGroups, taskBlockIndex, taskControlMapAndOffset,
                                     blockIdxOfTaskControlMap, barConsumptionReadyOrder);
            }

            _log.trace("Set barrier attributes for SHV in block {0}", taskBlockIndex);
            for (const auto& [queue, fifoExecGroups] : execGroupAnalysis.getActShvExecutionGroups()) {
                setBarrierAttributes(ctx, barrierInfo, fifoExecGroups, taskBlockIndex, taskControlMapAndOffset,
                                     blockIdxOfTaskControlMap, barConsumptionReadyOrder);
            }
        }

        verifyCleanAfterAttribute(barrierInfo, config::ExecutorKind::DPU);
        verifyCleanAfterAttribute(barrierInfo, config::ExecutorKind::SHAVE_ACT);
        barrierInfo.clearAttributes();
    }

    auto& barrierSim = getAnalysis<VPURT::BarrierSimulator>();

    VPUX_THROW_WHEN(barrierSim.isDynamicBarriers(), "The pass should be called for static barriers only");

    if (mlir::failed(barrierSim.checkProducerCount(_log.nest()))) {
        signalPassFailure();
        return;
    }
    if (mlir::failed(barrierSim.checkProducerAndConsumerCount(_log.nest()))) {
        signalPassFailure();
        return;
    }
    // For the simulation to run correctly barriers need to be ordered
    // based on first barrier producer order
    if (mlir::failed(barrierSim.simulateBarriers(_log.nest()))) {
        _log.error("Barrier simulation failed");
        signalPassFailure();
        return;
    }
}

}  // namespace

//
// createBarrierSimulationPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createBarrierSimulationPass(Logger log) {
    return std::make_unique<BarrierSimulationPass>(log);
}
