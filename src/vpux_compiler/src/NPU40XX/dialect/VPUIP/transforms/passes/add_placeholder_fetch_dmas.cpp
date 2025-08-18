//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_ADDPLACEHOLDERFETCHDMAS
#define GEN_PASS_DEF_ADDPLACEHOLDERFETCHDMAS
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;
namespace {

//
//  AddPlaceholderFetchDMAsPass
//

using BlockRange = SmallVector<std::pair<size_t, size_t>>;

struct FetchDMAData {
    size_t insertionPoint;
    SmallVector<IndexType> consumes;
    SmallVector<IndexType> producesIn;
    VPUIP::FetchDMAAttr fetchDmaAttr;
};

class AddPlaceholderFetchDMAsPass final :
        public VPUIP::arch40xx::impl::AddPlaceholderFetchDMAsBase<AddPlaceholderFetchDMAsPass> {
public:
    explicit AddPlaceholderFetchDMAsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void planFetchDMAAndBarriersInsertionPerQueue(BlockRange& blockRange, ExecutionGroupList& executionGroup,
                                                  BarrierInfo& barrierInfo);
    void realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo);
    VPUIP::FetchDMAAttr getFetchDMAAttr(int64_t groupIdx, BarrierInfo& barrierInfo, size_t taskIndex);

private:
    // Will be initialized in safeRunOnFunc() or relevant function, this is done to suppress the UNINIT_CTOR
    // warning
    size_t _numAllTaskOps = 0;
    size_t _newBarrierIndex = 0;
    mlir::Operation* _bufferInsertionPoint = nullptr;
    mlir::Operation* _barrierInsertionPoint = nullptr;
    VPURT::TaskOp _firstTaskOp;

    SmallVector<FetchDMAData> _fetchDMAsToInsert;
    llvm::DenseMap<IndexType, std::pair<SmallVector<IndexType>, SmallVector<IndexType>>> _barrierAddConsumerProducerMap;

    SmallVector<VPURT::DeclareVirtualBarrierOp> _dummyBarriers;
    SmallVector<VPURT::TaskOp> _fetchDMAs;
};

// Function to get tile index for DPU/SHV Op
size_t getTileIndexForDpuOrShv(BarrierInfo& barrierInfo, size_t taskIdx) {
    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
    if (auto dmaOp = taskOp.getInnerTaskOpOfType<VPUIP::NNDMAOp>()) {
        VPUX_THROW("getTileIndexForDpuOrShv called for DMAOp {0}", taskOp);
    }

    if (auto swOp = taskOp.getInnerTaskOpOfType<VPUIP::SwKernelOp>()) {
        return swOp.getTileIndex().value_or(0);
    }

    auto taskQueueType = barrierInfo.getTaskQueueType(taskIdx);
    return taskQueueType.id;
}

// Function to get list index for DPU/SHV Op
size_t getListIndexForDpuOrShv(BarrierInfo& barrierInfo, size_t taskIdx) {
    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
    if (auto dmaOp = taskOp.getInnerTaskOpOfType<VPUIP::NNDMAOp>()) {
        VPUX_THROW("getListIndexForDpuOrShv called for DMAOp {0}", taskOp);
    }

    if (auto swOp = taskOp.getInnerTaskOpOfType<VPUIP::SwKernelOp>()) {
        return swOp.getListIndex().value_or(0);
    }

    // All DPU tasks are expected to be on list 0
    return 0;
}

VPURT::TaskOp createFetchDma(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                             BarrierInfo& barrierInfo, VPUIP::FetchDMAAttr fetchDMAData) {
    auto newDMA = createFetchDMA(builder, inputBuf, outputBuf, 0, {}, {}, fetchDMAData);
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
void AddPlaceholderFetchDMAsPass::realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo) {
    auto inBuffer = VPUIP::createDummyBuffer(builder, _bufferInsertionPoint);
    auto outBuffer = VPUIP::createDummyBuffer(builder, _bufferInsertionPoint);

    // Create as many dummy barriers as were indexed during scheduling
    for ([[maybe_unused]] size_t unused = 0; unused < _newBarrierIndex; ++unused) {
        auto newBarrierOp = createNewBarrier(builder, barrierInfo, _barrierInsertionPoint, nullptr, nullptr);
        _dummyBarriers.push_back(newBarrierOp);
    }

    for (const auto& [dummyDmaIndex, value] : _fetchDMAsToInsert | indexed) {
        auto insertionPointOp = barrierInfo.getTaskOpAtIndex(value.insertionPoint);
        // Ensure fetch DMAs for first 2 groups are always first in the list
        if (value.fetchDmaAttr.getExecGroupIdx().getValue().getSExtValue() < 2) {
            builder.setInsertionPoint(insertionPointOp);
        } else {
            builder.setInsertionPointAfter(insertionPointOp);
        }
        auto dummyDMA = createFetchDma(builder, inBuffer, outBuffer, barrierInfo, value.fetchDmaAttr);
        _fetchDMAs.push_back(dummyDMA);
    }

    // We have created the new DMAs and barriers, adjust dependencies
    for (const auto& [dummyDmaIndex, value] : _fetchDMAsToInsert | indexed) {
        SmallVector<size_t> realProducesIn;
        SmallVector<size_t> realConsumes;
        for (auto produce : value.producesIn) {
            auto realBarrierIdx = getIndexOfBarrier(produce, _dummyBarriers, barrierInfo);
            realProducesIn.push_back(realBarrierIdx);
        }
        for (auto consume : value.consumes) {
            auto realBarrierIdx = getIndexOfBarrier(consume, _dummyBarriers, barrierInfo);
            realConsumes.push_back(realBarrierIdx);
        }
        updateBarriersForDma(realConsumes, realProducesIn, _fetchDMAs[dummyDmaIndex], barrierInfo);
    }

    for (const auto& [indexType, value] : _barrierAddConsumerProducerMap) {
        auto realBarrierIdx = getIndexOfBarrier(indexType, _dummyBarriers, barrierInfo);
        for (auto consumer : value.first) {
            auto realTaskIdx = getIndexOfTask(consumer, _fetchDMAs, barrierInfo);
            barrierInfo.addConsumer(realBarrierIdx, realTaskIdx);
        }
        for (auto producer : value.second) {
            auto realTaskIdx = getIndexOfTask(producer, _fetchDMAs, barrierInfo);
            barrierInfo.addProducer(realBarrierIdx, realTaskIdx);
        }
    }
}

VPUIP::FetchDMAAttr AddPlaceholderFetchDMAsPass::getFetchDMAAttr(int64_t groupIdx, BarrierInfo& barrierInfo,
                                                                 size_t taskIndex) {
    auto ctx = &(getContext());
    auto taskQueueType = barrierInfo.getTaskQueueType(taskIndex);
    auto executorKindAttr = VPU::ExecutorKindAttr::get(ctx, taskQueueType.type);
    auto tileIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), getTileIndexForDpuOrShv(barrierInfo, taskIndex));
    auto listIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), getListIndexForDpuOrShv(barrierInfo, taskIndex));
    auto groupIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), groupIdx);
    return VPUIP::FetchDMAAttr::get(ctx, executorKindAttr, tileIdxAttr, listIdxAttr, groupIdxAttr);
}

// Insert barrier and FetchDMA for each group
// LastOfGrandParent --> NewBarrier1 --> FetchDMA --> NewBarrier2 --> LastOfParent/SyncTask
void AddPlaceholderFetchDMAsPass::planFetchDMAAndBarriersInsertionPerQueue(BlockRange& blockRange,
                                                                           ExecutionGroupList& executionGroup,
                                                                           BarrierInfo& barrierInfo) {
    // Always populate for whatever is available for first 2 groups
    for (size_t groupIdx = 0; groupIdx < std::min<size_t>(2, executionGroup.size()); ++groupIdx) {
        FetchDMAData fetchDMAData;
        auto insertionIndex = barrierInfo.getIndex(_firstTaskOp);
        fetchDMAData.insertionPoint = insertionIndex;
        fetchDMAData.fetchDmaAttr = getFetchDMAAttr(groupIdx, barrierInfo, executionGroup[groupIdx].front());
        _fetchDMAsToInsert.push_back(fetchDMAData);
    }

    // If less than 3 groups, skip the rest of the logic that depends on both being present
    if (executionGroup.size() < 3) {
        return;
    }

    size_t groupIdx = 2;
    auto grandParentGroup = executionGroup.front();
    auto parentGroup = executionGroup[1];
    auto travelingGroup = executionGroup[groupIdx];
    while (groupIdx < executionGroup.size()) {
        auto lastTaskGrandParentGroup = grandParentGroup[grandParentGroup.size() - 1];
        auto lastTaskParentGroup = parentGroup.back();
        auto dummyBarrierTwoConsumer = lastTaskParentGroup;

        FetchDMAData fetchDMAData;
        // If both tasks are in different blocks, we may need a sync task as consumer for PlaceholderFetchDMA
        if (!inSameTaskBlock(lastTaskParentGroup, lastTaskGrandParentGroup, blockRange)) {
            auto syncPoint = barrierInfo.getControlGraphSyncPoint(lastTaskGrandParentGroup);
            // lastTaskGrandParentGroup is NOT the sync task — safe to use sync as consumer
            if (lastTaskGrandParentGroup != syncPoint.value()) {
                dummyBarrierTwoConsumer = syncPoint.value();
            } else {
                // lastTaskGrandParentGroup IS the sync task
                auto blockInd1 = barrierInfo.getControlGraphBlockIndex(lastTaskGrandParentGroup);
                auto blockInd2 = barrierInfo.getControlGraphBlockIndex(lastTaskParentGroup);

                if (blockInd1 + 1 == blockInd2) {
                    // Tasks are in consecutive blocks — use parent directly as consumer
                    dummyBarrierTwoConsumer = lastTaskParentGroup;
                } else {
                    // Need next block's sync point as consumer
                    auto nextSync = barrierInfo.getNextBlockSyncPoint(lastTaskGrandParentGroup);
                    VPUX_THROW_UNLESS(nextSync.has_value(), "No next block sync point found for FetchDMA consumer");
                    dummyBarrierTwoConsumer = nextSync.value();
                }
            }
        }

        // 1. LastOfGrandParent → B1
        size_t barOneDummyIdx = _newBarrierIndex++;
        auto& producersToAdd = _barrierAddConsumerProducerMap[{barOneDummyIdx, Type::Dummy}].second;
        producersToAdd.push_back({lastTaskGrandParentGroup, Type::Real});

        // 2. B1 → FetchDMA → B2
        fetchDMAData.insertionPoint = lastTaskGrandParentGroup;
        fetchDMAData.consumes = {{barOneDummyIdx, Type::Dummy}};

        size_t barTwoDummyIdx = _newBarrierIndex++;
        fetchDMAData.producesIn = {{barTwoDummyIdx, Type::Dummy}};

        // 3. B2 → LastOfParent (or syncTask)
        auto& consumersToAdd = _barrierAddConsumerProducerMap[{barTwoDummyIdx, Type::Dummy}].first;
        consumersToAdd.push_back({dummyBarrierTwoConsumer, Type::Real});

        fetchDMAData.fetchDmaAttr = getFetchDMAAttr(groupIdx, barrierInfo, travelingGroup.front());
        _fetchDMAsToInsert.push_back(fetchDMAData);

        grandParentGroup = parentGroup;
        parentGroup = travelingGroup;

        ++groupIdx;
        if (groupIdx < executionGroup.size()) {
            travelingGroup = executionGroup[groupIdx];
        }
    }
}

void AddPlaceholderFetchDMAsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    mlir::OpBuilder builder(netFunc);

    // Identify existing position of DeclareBufferOp, will be used as insertion point
    // for new tasks that will be inserted in IR
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    _bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    auto barrierOps = netFunc.getOps<VPURT::DeclareVirtualBarrierOp>();
    _barrierInsertionPoint = !barrierOps.empty() ? *barrierOps.begin() : &netFunc.getBody().front().front();

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    _numAllTaskOps = barrierInfo.getNumOfTasks();

    // Get the blockRanges to check we don't add deps between blocks
    BlockRange blockRange;
    for (size_t blockIdx = 0; blockIdx < barrierInfo.getControlGraphBlockCount(); ++blockIdx) {
        auto [blockStartInd, blockEndInd] = barrierInfo.getControlGraphBlockTaskRange(
                blockIdx, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ true);
        blockRange.push_back({blockStartInd, blockEndInd});
    }

    // Build task queue type map for all queues in order to test paths between tasks on different FIFOs.
    barrierInfo.initializeTaskQueueTypeMap(
            {VPU::ExecutorKind::DMA_NN, VPU::ExecutorKind::DPU, VPU::ExecutorKind::SHAVE_ACT});
    barrierInfo.buildTaskQueueTypeMap();

    // Will have a map for each cluster along with task index of the task
    auto taskQueues = VPURT::getTaskOpQueues(netFunc, barrierInfo);

    VPURT::TaskQueueType fetchDmaQueueType;
    fetchDmaQueueType.type = VPU::ExecutorKind::DMA_NN;
    fetchDmaQueueType.id = getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR);
    // _firstTaskOp is used as insertion point for FetchDMAs for initial 2 execution groups
    // If we have any DMAs on supported port and channel then FetchDMAs must be placed before them
    // If we don't have any DMAs on suported port and channel we can just place FetchDMA before first TaskOp
    auto taskOps = netFunc.getOps<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOps.empty(), "Can not find TaskOp");

    _firstTaskOp = *taskOps.begin();
    if (!taskQueues[fetchDmaQueueType].empty()) {
        _firstTaskOp = barrierInfo.getTaskOpAtIndex(taskQueues[fetchDmaQueueType].front());
    }

    auto& execGroupAnalysis = getAnalysis<ExecutionGroupAnalysis>();
    execGroupAnalysis.logExecutionGroupTasks(_log);
    auto dpuGroups = execGroupAnalysis.getDPUExecutionGroups();
    auto swGroups = execGroupAnalysis.getActShvExecutionGroups();

    for (auto& [_, executionGroups] : dpuGroups) {
        planFetchDMAAndBarriersInsertionPerQueue(blockRange, executionGroups, barrierInfo);
    }
    for (auto& [_, executionGroups] : swGroups) {
        planFetchDMAAndBarriersInsertionPerQueue(blockRange, executionGroups, barrierInfo);
    }

    realizePlannedInsertions(builder, barrierInfo);
    finalizeBarrierInfo(barrierInfo, netFunc, _log);

    // Log the number of inserted FetchDMAs
    if (!_fetchDMAsToInsert.empty()) {
        _log.info("Inserted '{0}' FetchDMAs", _fetchDMAsToInsert.size());
    }
}

}  // namespace

//
// createAddPlaceholderFetchDMAsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createAddPlaceholderFetchDMAsPass(Logger log) {
    return std::make_unique<AddPlaceholderFetchDMAsPass>(log);
}
