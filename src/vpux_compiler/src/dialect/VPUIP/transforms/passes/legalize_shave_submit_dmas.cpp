//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <ostream>
#include <string>
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/shave.hpp"
namespace vpux::VPUIP {
#define GEN_PASS_DECL_LEGALIZESHAVESUBMITDMAS
#define GEN_PASS_DEF_LEGALIZESHAVESUBMITDMAS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

enum class DMAType { Fetch = 0, Skip = 1, Sync = 2 };
struct PlannedSkip {
    size_t tile;
    size_t list;
    size_t channel;
    VPUIP::SkipDMAAttr skipAttr;
    size_t descId;
};

struct DMAData {
    size_t insertionPoint;
    VPUIP::DmaChannelType channelType;
    SmallVector<VPURT::IndexType> consumes;
    SmallVector<VPURT::IndexType> producesIn;
    DMAType dmaType;
    VPUIP::SkipDMAAttr skipDmaAttr;
    VPUIP::FetchDMAAttr fetchDmaAttr;
};

struct PlannedInsertionsData {
    size_t newDmaIndex = 0;
    size_t newBarrierIndex = 0;
    mlir::Operation* bufferInsertionPoint = nullptr;
    mlir::Operation* barrierInsertionPoint = nullptr;

    SmallVector<DMAData> dmasToInsert;
    llvm::DenseMap<VPURT::IndexType, std::pair<SmallVector<VPURT::IndexType>, SmallVector<VPURT::IndexType>>>
            barrierAddConsumerProducerMap;
};

using TaskOpQueues = std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>;

//
//  LegalizeShaveSubmitDMAsPass
//

class LegalizeShaveSubmitDMAsPass final : public VPUIP::impl::LegalizeShaveSubmitDMAsBase<LegalizeShaveSubmitDMAsPass> {
public:
    explicit LegalizeShaveSubmitDMAsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void planLegalization(BarrierInfo& barrierInfo, const SmallVector<size_t>& shvTasksWithDma,
                          PlannedInsertionsData& preparedInsertions, Logger& log);
    void realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                  PlannedInsertionsData& preparedInsertions, Logger& log);

private:
    void safeRunOnFunc() final;
};

void dumpShvTasksWithDma(const SmallVector<size_t>& shvTasksWithDma, Logger& log) {
    log.trace("SHV TaskOps with DMAs found at the following task indices:");
    for (auto taskIdx : shvTasksWithDma) {
        log.trace(" - Task index {0}", taskIdx);
    }
}

void findShvTasksWithDma(SmallVector<size_t>& shvTasksWithDma, BarrierInfo& barrierInfo, Logger& log) {
    log.trace("Finding SHV TaskOps with DMAs and grouping them by logical task index");
    for (size_t taskIdx = 0; taskIdx < barrierInfo.getNumOfTasks(); taskIdx++) {
        if (barrierInfo.getTaskQueueType(taskIdx).type != config::ExecutorKind::SHAVE_ACT) {
            continue;
        }
        auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
        auto swKernelOp = mlir::cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
        if (!isSwKernelUsingDma(swKernelOp)) {
            continue;
        }
        shvTasksWithDma.push_back(taskIdx);
    }
}

// Returns a DMA which copies 0 len data from DDR to DDR
VPURT::TaskOp createDmaForGivenType(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                                    BarrierInfo& barrierInfo, const DMAData& dmaData) {
    // Create sync DMA based on queue type
    VPURT::TaskOp newDMA;
    switch (dmaData.dmaType) {
    case DMAType::Sync:
        newDMA = VPUIP::createSyncDMA(builder, inputBuf, outputBuf, 0, {}, {}, "shv_submit_sync_dma");
        break;
    case DMAType::Skip:
        newDMA = VPURT::createSkipDMA(builder, inputBuf, outputBuf, /*port=*/0, dmaData.skipDmaAttr,
                                      "shv_submit_skip_dma");
        break;
    case DMAType::Fetch:
        newDMA = VPURT::createFetchDMA(builder, inputBuf, outputBuf, 0, {}, {}, dmaData.fetchDmaAttr,
                                       "shv_submit_fetch_dma");
        break;
    default:
        VPUX_THROW("Unknown DMAType");
        break;
    }
    barrierInfo.addNewTaskOp(newDMA);
    return newDMA;
}

void finalizeBarrierInfo(BarrierInfo& barrierInfo, mlir::func::FuncOp netFunc, Logger& log) {
    // Update IR, verify schedules
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(netFunc);
}

// Once we know the insertion point of DMAs this function creates actual DMAs in IR while also keeps a map of [index,
// DMAOp This map is used later to refer to real DMAOp and get real task-index from barrierInfo
void LegalizeShaveSubmitDMAsPass::realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                           PlannedInsertionsData& preparedInsertions, Logger& log) {
    log.trace("Realizing planned insertions of DMAs and barriers in IR");
    auto toMemoryKind = [](VPUIP::DmaChannelType type) -> VPU::MemoryKind {
        switch (type) {
        case VPUIP::DmaChannelType::DDR:
            return VPU::MemoryKind::DDR;

        case VPUIP::DmaChannelType::CMX:
            return VPU::MemoryKind::CMX_NN;

        case VPUIP::DmaChannelType::NOT_SPECIFIED:
            VPUX_THROW("DmaChannelType::NOT_SPECIFIED is not valid here");

        default:
            VPUX_THROW("Unknown DmaChannelType");
        }
    };

    mlir::Value inBuffer;
    mlir::Value outBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint);

    SmallVector<VPURT::TaskOp> insertedDMAs;
    insertedDMAs.reserve(preparedInsertions.newDmaIndex);

    SmallVector<VPURT::DeclareVirtualBarrierOp> dummyBarriers;
    dummyBarriers.reserve(preparedInsertions.newBarrierIndex);

    // Create as many dummy barriers as were indexed during scheduling
    for (size_t i = 0; i < preparedInsertions.newBarrierIndex; ++i) {
        auto newBarrierOp = VPURT::createNewBarrier(builder, barrierInfo, preparedInsertions.barrierInsertionPoint,
                                                    nullptr, nullptr);
        dummyBarriers.push_back(newBarrierOp);
    }

    // This holds the buffer for each channel type which is used in newly created DMAs. We create one buffer per channel
    // type and reuse it for all DMAs of that channel type
    DenseMap<VPUIP::DmaChannelType, mlir::Value> taskQueueBufferMap;
    for (const auto& [dummyDmaIndex, value] : preparedInsertions.dmasToInsert | indexed) {
        auto insertionPointOp = barrierInfo.getTaskOpAtIndex(value.insertionPoint);
        builder.setInsertionPoint(insertionPointOp);

        auto it = taskQueueBufferMap.find(value.channelType);
        if (it != taskQueueBufferMap.end()) {
            log.trace("Reusing existing buffer for channel type {0}", value.channelType);
            inBuffer = it->second;
        } else {
            log.trace("Creating dummy buffer for channel type {0}", value.channelType);
            inBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint,
                                                toMemoryKind(value.channelType));
            taskQueueBufferMap[value.channelType] = inBuffer;
        }

        auto dummyDMA = createDmaForGivenType(builder, inBuffer, outBuffer, barrierInfo, value);
        insertedDMAs.push_back(dummyDMA);
    }

    // We have created the new DMAs and barriers, adjust dependencies
    for (const auto& [dummyDmaIndex, value] : preparedInsertions.dmasToInsert | indexed) {
        SmallVector<size_t> realProducesIn;
        SmallVector<size_t> realConsumes;

        for (auto produce : value.producesIn) {
            auto realBarrierIdx = getIndexOfBarrier(produce, dummyBarriers, barrierInfo);
            log.trace("Add dependency between task at index {0} and barrier at index {1}", dummyDmaIndex,
                      realBarrierIdx);
            realProducesIn.push_back(realBarrierIdx);
        }
        for (auto consume : value.consumes) {
            auto realBarrierIdx = getIndexOfBarrier(consume, dummyBarriers, barrierInfo);
            log.trace("Add dependency between task at index {0} and barrier at index {1}", dummyDmaIndex,
                      realBarrierIdx);
            realConsumes.push_back(realBarrierIdx);
        }
        updateBarriersForDma(realConsumes, realProducesIn, insertedDMAs[dummyDmaIndex], barrierInfo);
    }

    for (const auto& [indexType, value] : preparedInsertions.barrierAddConsumerProducerMap) {
        auto realBarrierIdx = getIndexOfBarrier(indexType, dummyBarriers, barrierInfo);
        for (auto consumer : value.first) {
            auto realTaskIdx = getIndexOfTask(consumer, insertedDMAs, barrierInfo);
            log.trace("Add dependency between barrier at index {0} and task at index {1}", realBarrierIdx, realTaskIdx);
            barrierInfo.addConsumer(realBarrierIdx, realTaskIdx);
        }
        for (auto producer : value.second) {
            auto realTaskIdx = getIndexOfTask(producer, insertedDMAs, barrierInfo);
            log.trace("Add dependency between barrier at index {0} and task at index {1}", realBarrierIdx, realTaskIdx);
            barrierInfo.addProducer(realBarrierIdx, realTaskIdx);
        }
    }
}
/*
This function plans legalization of SHV tasks submitting DMAs by inserting required Skip along with Fetch DMAs and
barriers to ensure that all fetches are completed and DMA Descriptors for Skip are in CMX before SHV starts executing.

-> Fetch(Skip_list0_DDR) -|
-> Fetch(Skip_list1_DDR) -|-> BAR -> |-> SyncDMA(DDR) -> SkipDMA(list0_DDR), SkipDMA(list1_DDR)
-> Fetch(Skip_list0_CMX) -|          |-> SyncDMA(CMX) -> SkipDMA(list0_CMX), SkipDMA(list1_CMX)
-> Fetch(Skip_list1_CMX) -|          |
                                     |------------------------> SHV[list0](withDMA)
                                     |------------------------> SHV[list1](withDMA)

Note: The pass currently ONLY legalizes for single DMA port/engine, to support multiple ports/engines the logic needs
to be extended using same fundamentals
*/

void LegalizeShaveSubmitDMAsPass::planLegalization(BarrierInfo& barrierInfo, const SmallVector<size_t>& shvTasksWithDma,
                                                   PlannedInsertionsData& preparedInsertions, Logger& log) {
    log.trace("Planning legalization of SHV submit DMAs");
    std::map<size_t, SmallVector<size_t>> shvGroups;

    // Group by logical task index
    for (auto taskIdx : shvTasksWithDma) {
        auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
        auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
        VPUX_THROW_UNLESS(swKernelOp != nullptr, "Expected SwKernelOp inside SHV TaskOp");

        const auto logicalTaskAttr = swKernelOp->getAttr(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);
        VPUX_THROW_UNLESS(
                logicalTaskAttr != nullptr,
                "SwKernelOp at index '{0}' is missing '{1}' IntegerAttr required by LegalizeShaveSubmitDMAsPass",
                taskIdx, VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);

        const auto logicalIdx = mlir::cast<mlir::IntegerAttr>(logicalTaskAttr).getValue().getSExtValue();
        shvGroups[logicalIdx].push_back(taskIdx);
    }

    if (_log.isActive(LogLevel::Trace)) {
        dumpShvTasksWithDma(shvTasksWithDma, log);
    }

    log.trace("Planning insertions for SHV tasks with DMAs. Number of logical groups: {0}", shvGroups.size());
    const auto channelDDR = static_cast<size_t>(VPUIP::DmaChannelType::DDR);
    const auto channelCMX = static_cast<size_t>(VPUIP::DmaChannelType::CMX);
    size_t descId = 0;

    // Go over SHV tasks with same logicalTaskIndex and create DMAs and barriers for them. The plan is to create one
    // barrier per logical task, then add fetch DMAs as producers to the barrier and SHV tasks as consumers. This way we
    // ensure that all fetches are done before SHV starts executing.
    for (auto& [logicalIdx, taskIndices] : shvGroups) {
        auto earliestIdx = taskIndices.front();

        // One barrier per logical task
        auto groupBarrierIdx = preparedInsertions.newBarrierIndex++;

        // Temporary storage to enforce ordering later
        SmallVector<DMAData> plannedFetches;
        SmallVector<DMAData> plannedSkips;
        mlir::OpBuilder builder(barrierInfo.getTaskOpAtIndex(earliestIdx));

        for (auto taskIdx : taskIndices) {
            log.trace("Planning Fetches and skips for logical task index {0} with {1} SHV tasks", logicalIdx,
                      taskIndices.size());

            SmallVector<mlir::Attribute> descIdAttrs;
            auto swTaskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
            auto swOp = mlir::cast<VPUIP::SwKernelOp>(swTaskOp.getInnerTaskOp());
            auto taskQueueType = VPURT::getTaskQueueType(swTaskOp, false);

            auto tile = VPURT::getTileIndexForDpuOrShv(swTaskOp, taskQueueType);
            auto list = VPURT::getListIndexForDpuOrShv(swTaskOp);

            for (auto channel : {channelDDR, channelCMX}) {
                const size_t pairDescId = descId++;
                descIdAttrs.push_back(builder.getI64IntegerAttr(pairDescId));

                DMAData fetch;
                preparedInsertions.newDmaIndex++;
                fetch.insertionPoint = earliestIdx;
                fetch.channelType = static_cast<VPUIP::DmaChannelType>(channelDDR);
                fetch.dmaType = DMAType::Fetch;
                fetch.fetchDmaAttr =
                        VPURT::getFetchDMAAttr(logicalIdx, barrierInfo, taskIdx, tile, list, pairDescId, true);

                fetch.producesIn = {{groupBarrierIdx, VPURT::Type::Dummy}};
                plannedFetches.push_back(fetch);

                DMAData skip;
                preparedInsertions.newDmaIndex++;
                skip.insertionPoint = earliestIdx;
                skip.channelType = static_cast<VPUIP::DmaChannelType>(channel);
                skip.dmaType = DMAType::Skip;
                skip.skipDmaAttr = VPURT::getSkipDMAAttr(barrierInfo, taskIdx, logicalIdx, pairDescId);

                plannedSkips.push_back(skip);
            }
            swOp.setSkipDescIdsAttr(builder.getArrayAttr(descIdAttrs));
        }

        // Insert Fetches in preparedInsertions first so that they are guaranteed to be before Sync and Skip DMAs in IR
        for (auto& fetch : plannedFetches) {
            preparedInsertions.dmasToInsert.push_back(fetch);
        }

        // Insert Sync DMAs for both channels. These DMAs will be used to update the barrier and ensure that SHV waits
        // for fetches to complete and descriptors to be ready in CMX before starting execution
        log.trace("Planning Sync DMAs for logical task index {0} ", logicalIdx);
        DMAData syncDDR;
        preparedInsertions.newDmaIndex++;
        syncDDR.insertionPoint = earliestIdx;
        syncDDR.channelType = VPUIP::DmaChannelType::DDR;
        syncDDR.dmaType = DMAType::Sync;
        syncDDR.consumes = {{groupBarrierIdx, VPURT::Type::Dummy}};
        preparedInsertions.dmasToInsert.push_back(syncDDR);

        DMAData syncCMX;
        preparedInsertions.newDmaIndex++;
        syncCMX.insertionPoint = earliestIdx;
        syncCMX.channelType = VPUIP::DmaChannelType::CMX;
        syncCMX.dmaType = DMAType::Sync;
        syncCMX.consumes = {{groupBarrierIdx, VPURT::Type::Dummy}};
        preparedInsertions.dmasToInsert.push_back(syncCMX);

        // Technically we don't care what order skip loop is running however we need to be deterministic for
        // debuggability and testing purposes
        // Skip Tile 0 List 0->Tile 0 List 1-> Tile 1 List 0 -> Tile 1 List 1
        log.trace("Reordering planned Skip DMAs for logical task index {0} to ensure deterministic order in IR",
                  logicalIdx);
        llvm::sort(plannedSkips, [](const DMAData& a, const DMAData& b) {
            auto aAttr = a.skipDmaAttr;
            auto bAttr = b.skipDmaAttr;

            const auto aTile = aAttr.getAssociatedTileIdx().getValue().getSExtValue();
            const auto aList = aAttr.getAssociatedListIdx().getValue().getSExtValue();

            const auto bTile = bAttr.getAssociatedTileIdx().getValue().getSExtValue();
            const auto bList = bAttr.getAssociatedListIdx().getValue().getSExtValue();

            return std::tie(aTile, aList) < std::tie(bTile, bList);
        });

        for (auto& skip : plannedSkips) {
            preparedInsertions.dmasToInsert.push_back(skip);
        }

        // Ask SHV to wait for fetches by adding them as dependencies to the barrier, and then add SHV tasks as
        // consumers of the barrier as well
        log.trace("Add barrier dependencies for logical task index {0} ", logicalIdx);
        for (auto taskIdx : taskIndices) {
            log.trace("dependency for task {0} on barrier {1} added", taskIdx, groupBarrierIdx);
            auto& consumersToAdd =
                    preparedInsertions.barrierAddConsumerProducerMap[{groupBarrierIdx, VPURT::Type::Dummy}].first;
            consumersToAdd.push_back({taskIdx, VPURT::Type::Real});
        }
    }
}

void LegalizeShaveSubmitDMAsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    mlir::OpBuilder builder(netFunc);
    PlannedInsertionsData preparedInsertions;

    // Identify existing position of DeclareBufferOp, will be used as insertion point
    // for new tasks that will be inserted in IR
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    preparedInsertions.bufferInsertionPoint =
            !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    auto barrierOps = netFunc.getOps<VPURT::DeclareVirtualBarrierOp>();
    preparedInsertions.barrierInsertionPoint =
            !barrierOps.empty() ? *barrierOps.begin() : &netFunc.getBody().front().front();

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    // Build task queue type map for all queues in order to test paths between tasks on different FIFOs.
    barrierInfo.buildTaskQueueTypeMap();

    // Store information about SHV tasks which can submit DMA ops
    SmallVector<size_t> shvTasksWithDma;
    findShvTasksWithDma(shvTasksWithDma, barrierInfo, _log);
    planLegalization(barrierInfo, shvTasksWithDma, preparedInsertions, _log);
    realizePlannedInsertions(builder, barrierInfo, preparedInsertions, _log);
    finalizeBarrierInfo(barrierInfo, netFunc, _log);
}
}  // namespace

//
// createLegalizeShaveSubmitDMAsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createLegalizeShaveSubmitDMAsPass(Logger log) {
    return std::make_unique<LegalizeShaveSubmitDMAsPass>(log);
}
