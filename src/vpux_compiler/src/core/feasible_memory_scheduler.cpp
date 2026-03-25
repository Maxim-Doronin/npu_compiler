//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/async_dialect_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/async_dialect_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

using namespace vpux;
using operationIdxType = FeasibleMemoryScheduler::operationIdxType;
//
// Feasible Memory Scheduler
//

// This class will try to produce a feasible memory schedule based on the dependency map provided from
// AsyncDepsInfo and use the LinearScan class to allocate the resources.
// Data and Compute ops, where Data ops are operations moving data to CMX are distinguished in order to
// follow the scheduling of Compute ops along with their dependencies (Data ops). This optimizes CMX usage,
// and allows for feasible CMX schedule to be generated.
// The graph is iterated topologically based on the dependencies from input to output(s).
// In init() ready lists will be populated using operations without dependencies.
// In schedulingLoop() there are two possible scenarios:
// 1. Scheduling the next earliest operation from the start cycle heap, and adding it to the op output table.
// 2. Un-scheduling operations: freeing CMX space and updating dependencies, creating new ready
//      operations which will be allocated at the next available cycle.

FeasibleMemoryScheduler::FeasibleMemoryScheduler(VPU::MemoryKind memKind, VPU::MemoryKind secondLvlMemKind,
                                                 MemLiveRangeInfo& liveRangeInfo, AsyncDepsInfo& depsInfo, Logger log,
                                                 LinearScan<mlir::Value, LinearScanHandler>& scan,
                                                 config::ArchKind arch, VPUNN::VPUDevice vpuDevice,
                                                 std::shared_ptr<VPUNN::VPUCostModel> costModel,
                                                 int64_t nceClusterCount, int64_t dmaCount,
                                                 bool enableScheduleStatistics, bool optimizeFragmentation,
                                                 bool activelySpillForPrefetching, const ComputeRegionVec& loopRegions)
        : _log(log),
          _memKind(memKind),
          _secondLvlMemKind(secondLvlMemKind),
          _liveRangeInfo(liveRangeInfo),
          _depsInfo(depsInfo),
          _scan(scan),
          _archKind(arch),
          _vpuDevice(vpuDevice),
          _costModel(std::move(costModel)),
          _nceClusterCount(nceClusterCount),
          _numDMAPorts(dmaCount),
          _enableScheduleStatistics(enableScheduleStatistics),
          _optimizeFragmentation(optimizeFragmentation),
          _activelySpillForPrefetching(activelySpillForPrefetching),
          _loopRegions(loopRegions) {
    _log.setName("feasible-memory-scheduler-allocator");

    auto dmaChannels = getDMAChannelsWithIndependentLinkAgents(arch);
    for (auto dmaChannel : dmaChannels) {
        QueueType queueType;
        queueType.execKind = config::ExecutorKind::DMA_NN;
        queueType.id = getDMAQueueIdEncoding(dmaChannel);
        _executorPipelines[queueType].assign(dmaCount, 1);
    }

    buildLoopOperationIndices();
}

/// Build operation index sets for loop region scheduling.
///
/// This function scans all loop regions and categorizes their operations into two sets:
/// 1. _loopRegionInd: Contains operation indices that belong to loop bodies and require
///    special loop scheduling (ping-pong buffer allocation). These operations are excluded
///    from normal prefetching and compute scheduling to ensure they are handled by the
///    dedicated loop scheduling logic in scheduleLoopRegions().
///
/// 2. _loopPrefetchInd: Contains DATA_IN operation indices within loops. These are input
///    data transfer operations (typically DMAs) that can still be prefetched during normal
///    scheduling, as prefetching input data doesn't interfere with the loop's ping-pong
///    buffer management strategy.
///
/// Called once during scheduler initialization before the main scheduling loop begins.
void FeasibleMemoryScheduler::buildLoopOperationIndices() {
    for (const auto& computeRegion : _loopRegions) {
        // Skip non-loop regions (regions without loop scheduling requirements)
        if (computeRegion.getLoopType() == LoopType::None) {
            continue;
        }
        _log.trace("{0}", computeRegion);

        // Iterate through all loop iterations (loop bodies) in this compute region
        for (const auto& loop : computeRegion.schedulingLoop->loopBodies) {
            // Categorize each operation in the loop body
            for (const auto& alloc : loop) {
                if (alloc.allocationType == AllocationType::DATA_IN) {
                    // DATA_IN operations (input DMAs) can be prefetched normally
                    // as they bring data into CMX before the loop iteration starts
                    _loopPrefetchInd.insert(alloc.opIdx);
                    continue;
                }
                // All other operations (COMPUTE, DATA_OUT, etc.) require special
                // loop scheduling with ping-pong buffer management
                _loopRegionInd.insert(alloc.opIdx);
            }
        }
    }
}

struct FeasibleMemoryScheduler::ScheduleStateSnapshot {
    FeasibleMemoryScheduler& scheduler;
    bool captured = false;
    bool finalized = false;

    // Captured state variables from the scheduler.
    decltype(scheduler._readyComputeOps) readyComputeOps;
    LinearScan<mlir::Value, LinearScanHandler> scan;
    decltype(scheduler._executorPipelines) executorPipelines;
    decltype(scheduler._cycleBeginHeap) cycleBeginHeap;
    decltype(scheduler._readyDataOps) readyDataOps;
    decltype(scheduler._readySpilledOps) readySpilledOps;
    decltype(scheduler._spillBufferMap) spillBufferMap;

    // Constructor to optionally capture the scheduler's state.
    ScheduleStateSnapshot(FeasibleMemoryScheduler& sched, bool needCapture): scheduler(sched), scan(sched._scan) {
        if (!needCapture) {
            return;
        }
        captured = true;  // Mark the state as captured.
        readyComputeOps = scheduler._readyComputeOps;
        scan = scheduler._scan;
        executorPipelines = scheduler._executorPipelines;
        cycleBeginHeap = scheduler._cycleBeginHeap;
        readyDataOps = scheduler._readyDataOps;
        readySpilledOps = scheduler._readySpilledOps;
        spillBufferMap = scheduler._spillBufferMap;
    }

    // Restores the scheduler's state to the captured snapshot.
    void rollback() {
        if (!captured || finalized) {
            return;  // If the state was not captured or already finalized, do nothing.
        }
        scheduler._readyComputeOps = std::move(readyComputeOps);
        scheduler._scan = std::move(scan);
        scheduler._executorPipelines = std::move(executorPipelines);
        scheduler._cycleBeginHeap = std::move(cycleBeginHeap);
        scheduler._readyDataOps = std::move(readyDataOps);
        scheduler._readySpilledOps = std::move(readySpilledOps);
        scheduler._spillBufferMap = std::move(spillBufferMap);
        finalized = true;
    }
};

bool compareHeapOrderWhenCycleMatch(const FeasibleMemoryScheduler::HeapElement& a,
                                    const FeasibleMemoryScheduler::HeapElement& b) {
    if (a.isPrefetched() && !b.isPrefetched()) {
        return true;
    }
    if (!a.isPrefetched() && b.isPrefetched()) {
        return false;
    }
    if (a.spillBuffer_ != nullptr && b.spillBuffer_ != nullptr && a.spillBuffer_ != b.spillBuffer_) {
        return ValueOrderCmp::compare(a.spillBuffer_, b.spillBuffer_);
    }
    return a.op_ < b.op_;
}

// Sort heap by earliest begin cycle
bool FeasibleMemoryScheduler::CycleBeginMinHeapOrdering::operator()(const HeapElement& a, const HeapElement& b) const {
    if (a.cycleBegin_ != b.cycleBegin_) {
        return a.cycleBegin_ < b.cycleBegin_;
    }
    return compareHeapOrderWhenCycleMatch(a, b);
}

// Sort heap by earliest end cycle
bool FeasibleMemoryScheduler::CycleEndMinHeapOrdering::operator()(const HeapElement& a, const HeapElement& b) const {
    if (a.cycleEnd_ != b.cycleEnd_) {
        return a.cycleEnd_ < b.cycleEnd_;
    }
    return compareHeapOrderWhenCycleMatch(a, b);
}

void FeasibleMemoryScheduler::updateBufferCycleUseAndProducer(size_t opIdx, size_t opCycleEnd, const mlir::Value buffer,
                                                              bool isNewProducer) {
    // update buffer producer
    if (isNewProducer) {
        if (_bufferProducer.find(buffer) == _bufferProducer.end()) {
            _newBufferProducersFromScheduleComputeOps.insert(buffer);
        }

        if (_originalBufferProducersFromScheduleComputeOps.find(buffer) ==
            _originalBufferProducersFromScheduleComputeOps.end()) {
            _originalBufferProducersFromScheduleComputeOps[buffer] = _bufferProducer[buffer];
        }

        _bufferProducer[buffer] = opIdx;
    }
    // update last cycle use of buffer
    auto bufferUseCycleEnd = _bufferLastCycleUse.find(buffer);
    if (bufferUseCycleEnd != _bufferLastCycleUse.end()) {
        if (_originalBufferLastCycleUseFromScheduleComputeOps.find(buffer) ==
            _originalBufferLastCycleUseFromScheduleComputeOps.end()) {
            _originalBufferLastCycleUseFromScheduleComputeOps[buffer] = bufferUseCycleEnd->second;
        }
        bufferUseCycleEnd->second = std::max(bufferUseCycleEnd->second, opCycleEnd);
    } else {
        _bufferLastCycleUse[buffer] = opCycleEnd;
        _newBufferLastCycleUsesFromScheduleComputeOps.insert(buffer);
    }
}

void FeasibleMemoryScheduler::pushToCycleBeginHeap(const HeapElement& elem) {
    _cycleBeginHeap.insert(elem);
    // store as writer of output buffers
    if (elem.isSpillReadOp()) {
        updateBufferCycleUseAndProducer(elem.op_, elem.cycleEnd_, elem.spillBuffer_, true);
    } else if (elem.isOriginalOp()) {
        const auto execOp = _depsInfo.getExecuteOpAtIndex(elem.op_);
        for (auto& buffer : _liveRangeInfo.getOutputBuffers(execOp)) {
            updateBufferCycleUseAndProducer(elem.op_, elem.cycleEnd_, buffer, true);
        }
        for (auto& buffer : _liveRangeInfo.getInputBuffers(execOp)) {
            updateBufferCycleUseAndProducer(elem.op_, elem.cycleEnd_, buffer);
        }
    }
    insertInOpIdxCycleEndMap(elem.op_, elem.cycleEnd_);
}

size_t FeasibleMemoryScheduler::findMinScheduledQueueCycle() {
    // for all scheduled ops find the minimal queue cycle end
    size_t targetCycleEnd = std::numeric_limits<size_t>::max();
    std::map<QueueType, size_t> queueMinCycleEnd;
    for (const auto& op : _cycleEndHeap) {
        for (auto execInst : op.executorInstanceMask_.set_bits()) {
            targetCycleEnd = std::min(targetCycleEnd, _executorPipelines[op.queueType_][execInst]);
        }
    }
    return targetCycleEnd;
}

void FeasibleMemoryScheduler::moveFromCycleBeginToCycleEndHeap() {
    // move ops from cycle begin heap to cycle end heap
    for (auto& nextOp : _cycleBeginHeap) {
        _log.nest(2).trace("Move opIdx '{0}'", nextOp.op_);
        // add op to ScheduledOpVec
        populateScheduledOps(nextOp);
        // move to cycle end heap
        _cycleEndHeap.insert(nextOp);
        // decrease outputs if output operation scheduled
        if (_outputOps.find(nextOp.op_) != _outputOps.end()) {
            _outputOps.erase(nextOp.op_);
        }
    }

    _cycleBeginHeap.clear();
}

config::ExecutorKind FeasibleMemoryScheduler::getExecutorType(operationIdxType opIdx) {
    if (_spillBufferMap.find(opIdx) != _spillBufferMap.end()) {
        // spilled operation using DMAs for relocation
        return config::ExecutorKind::DMA_NN;
    }
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    return vpux::VPUIP::getExecutorType(execOp);
}

FeasibleMemoryScheduler::QueueType FeasibleMemoryScheduler::getQueueType(operationIdxType opIdx) {
    VPUX_THROW_WHEN(_spillBufferMap.find(opIdx) != _spillBufferMap.end(),
                    "Function does not support spilled operations, opIdx - '{0}'", opIdx);

    QueueType queueType;
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (execOp->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
        queueType.execKind = VPUIP::VPUIPDialect::getExecutorKind(execOp);

        if (auto dmaTask = VPUIP::getDmaTypeOp(execOp)) {
            queueType.id = getDMAQueueIdEncoding(dmaTask.getChannelType());
        }
        return queueType;
    }
    // for now treat all other executors as DPU - same as previous implementation
    queueType.execKind = config::ExecutorKind::DPU;
    return queueType;
}

// When getting number of ports needed for a task executing on DMA, this
// function determines if based on buffer type execution would require
// multiple ports
bool areMultipleDmaPortsNeeded(mlir::Value buffer) {
    if (auto distType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(buffer.getType())) {
        auto mode = distType.getDistribution().getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
            return true;
        }
    }
    return false;
}

// TODO: In future it might be desired to create some utility functions to gather information about
// the number of executors given operation requires
size_t FeasibleMemoryScheduler::getOpDemandForExecutorsInstances(operationIdxType opIdx, QueueType queueType) {
    auto numOfExecutors = _executorPipelines[queueType].size();
    VPUX_THROW_WHEN(numOfExecutors == 0, "No executor of given type {0} and id {1}", queueType.execKind, queueType.id);
    if (numOfExecutors < 2) {
        return 1;
    }

    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);

    // Current only for DMA tasks:
    // Check if operation works on DistributedBuffers with SEGMENTED mode. In such case
    // such DMA will be later split into per-cluster DMA tasks (unroll-distributed-ops pass).
    // Here assume that this operation will use all executors
    if (queueType.execKind == config::ExecutorKind::DMA_NN) {
        const auto usedBufs = _liveRangeInfo.getUsedBuffers(execOp);
        for (auto& buffer : usedBufs) {
            if (areMultipleDmaPortsNeeded(buffer)) {
                return numOfExecutors;
            }
        }
    }

    return 1;
}

size_t FeasibleMemoryScheduler::getBufferDemandForExecutorsInstances(mlir::Value buffer, QueueType queueType) {
    auto numOfExecutors = _executorPipelines[queueType].size();
    if (numOfExecutors < 2) {
        return 1;
    }

    // Current only for DMA tasks:
    // Check if operation works on DistributedBuffers with SEGMENTED mode. In such case
    // such DMA will be later split into per-cluster DMA tasks. Here assume that this operation
    // will use all executors
    if (queueType.execKind == config::ExecutorKind::DMA_NN) {
        if (areMultipleDmaPortsNeeded(buffer)) {
            return numOfExecutors;
        }
    }

    return 1;
}

llvm::BitVector FeasibleMemoryScheduler::getExecutorInstanceMask(size_t numOfNeededInstances, QueueType queueType) {
    auto numOfAllInstances = _executorPipelines[queueType].size();

    VPUX_THROW_UNLESS(numOfAllInstances > 0, "No available instances of given queue type");
    VPUX_THROW_UNLESS(numOfNeededInstances == 1 || numOfNeededInstances == numOfAllInstances,
                      "Number of needed executors ('{0}') is different then number of all instances of executor "
                      "('{1}'). This is not "
                      "yet supported",
                      numOfNeededInstances, numOfAllInstances);

    llvm::BitVector executorMask(checked_cast<uint32_t>(numOfAllInstances));

    if (queueType.execKind == config::ExecutorKind::DMA_NN) {
        if (numOfNeededInstances == 1) {
            // Find the executor with lowest cycle
            size_t indexMin = 0;
            size_t cycleMin = std::numeric_limits<size_t>::max();
            for (size_t i = 0; i < numOfAllInstances; i++) {
                if (_executorPipelines[queueType][i] < cycleMin) {
                    indexMin = i;
                    cycleMin = _executorPipelines[queueType][i];
                }
            }

            return executorMask.set(checked_cast<uint32_t>(indexMin));
        } else {
            return executorMask.set(0, checked_cast<uint32_t>(numOfAllInstances));
        }
    }

    return executorMask.set(0);
}

llvm::BitVector FeasibleMemoryScheduler::getExecutorInstanceMaskForOp(operationIdxType opIdx, QueueType queueType) {
    // TODO: If executor is configured in the operation read it directly from
    // operation async.execute. Currently this is not needed but in future
    // might be useful in case task distribution is performed by some earlier pass

    auto numOfNeededInstances = getOpDemandForExecutorsInstances(opIdx, queueType);

    return getExecutorInstanceMask(numOfNeededInstances, queueType);
}

llvm::BitVector FeasibleMemoryScheduler::getExecutorInstanceMaskForBuffer(mlir::Value buffer, QueueType queueType) {
    auto numOfNeededInstances = getBufferDemandForExecutorsInstances(buffer, queueType);

    return getExecutorInstanceMask(numOfNeededInstances, queueType);
}

FeasibleMemoryScheduler::QueueAndCycleType FeasibleMemoryScheduler::getCurrentCycleAndExecutorInstanceMask(
        operationIdxType opIdx, size_t depEndCycle) {
    auto queueType = getQueueType(opIdx);
    auto executorInstanceMask = getExecutorInstanceMaskForOp(opIdx, queueType);
    VPUX_THROW_WHEN(executorInstanceMask.none(), "No executor instance found");

    size_t earliestBeginCycle = depEndCycle;
    for (auto instIndex : executorInstanceMask.set_bits()) {
        earliestBeginCycle = std::max(earliestBeginCycle, _executorPipelines[queueType][instIndex]);
    }

    // check if operation cycle begin delayed by dependencies
    for (const auto& dep : _depsInfo.getOpDeps(opIdx)) {
        earliestBeginCycle = std::max(earliestBeginCycle, _opIdxEndCycleMap[dep]);
    }
    return QueueAndCycleType{queueType, std::move(executorInstanceMask), earliestBeginCycle};
}

FeasibleMemoryScheduler::QueueAndCycleType FeasibleMemoryScheduler::getCurrentCycleAndExecutorInstanceMaskForSpill(
        mlir::Value buffer, EOpType spillType, size_t depEndCycle) {
    QueueType queueType;
    queueType.execKind = config::ExecutorKind::DMA_NN;
    if (spillType == EOpType::IMPLICIT_SPILL_READ_OP) {
        queueType.id = getDMAQueueIdEncoding(_secondLvlMemKind, _archKind);
    } else {
        queueType.id = getDMAQueueIdEncoding(_memKind, _archKind);
    }

    auto executorInstanceMask = getExecutorInstanceMaskForBuffer(buffer, queueType);

    VPUX_THROW_WHEN(executorInstanceMask.none(), "No executor instance found");

    size_t earliestBeginCycle = depEndCycle;
    for (auto instIndex : executorInstanceMask.set_bits()) {
        earliestBeginCycle = std::max(earliestBeginCycle, _executorPipelines[queueType][instIndex]);
    }

    return QueueAndCycleType{queueType, std::move(executorInstanceMask), earliestBeginCycle};
}

void FeasibleMemoryScheduler::updateCurrentCycleForExecutor(QueueType queueType, llvm::BitVector executorInstanceMask,
                                                            size_t nextAvailableCycle) {
    for (auto execInst : executorInstanceMask.set_bits()) {
        _executorPipelines[queueType][execInst] = nextAvailableCycle;
    }
}

void FeasibleMemoryScheduler::alignExecutors(size_t nextAvailableCycle) {
    for (auto& pipeline : _executorPipelines) {
        auto numOfInst = pipeline.second.size();
        for (size_t i = 0; i < numOfInst; i++) {
            pipeline.second[i] = std::max(pipeline.second[i], nextAvailableCycle);

            std::string executorInstanceInfo = "";

            if (pipeline.first.execKind == config::ExecutorKind::DMA_NN) {
                auto channelTypeAsString = VPUIP::getDMAChannelTypeAsString(pipeline.first.id, _archKind);
                if (channelTypeAsString.size() > 0) {
                    executorInstanceInfo += "_" + channelTypeAsString;
                }
            }

            if (numOfInst > 1) {
                executorInstanceInfo += " [" + std::to_string(i) + "]";
            }

            _log.nest().trace("Aligning executor pipeline {0}{1} = {2}", pipeline.first.execKind, executorInstanceInfo,
                              pipeline.second[i]);
        }
    }
}

size_t FeasibleMemoryScheduler::spilledOperationCycleCost(mlir::Value spilledBuffer) {
    if (_spillBufferCycleCost.find(spilledBuffer) != _spillBufferCycleCost.end()) {
        // reuse calculated cycles
        return _spillBufferCycleCost[spilledBuffer];
    }
    // get and store cost of buffer spill
    _spillBufferCycleCost[spilledBuffer] =
            getDMACost(spilledBuffer, spilledBuffer, _archKind, _vpuDevice, _costModel, _numDMAPorts);
    return _spillBufferCycleCost[spilledBuffer];
}

size_t FeasibleMemoryScheduler::operationCycleCost(operationIdxType opIdx) {
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (!execOp->hasAttr(cycleCostAttrName)) {
        // operations without cycle cost will have cycle cost = 1
        _log.trace("async.exec {0} has no cycle cost attribute {1}", execOp->getLoc(), cycleCostAttrName);
        return 1;
    }

    return checked_cast<size_t>(
            mlir::cast<mlir::IntegerAttr>(execOp->getAttr(cycleCostAttrName)).getValue().getSExtValue());
}

bool FeasibleMemoryScheduler::isDataOp(operationIdxType opIdx) {
    // Operations moving data to CMX are considered data ops. All others are
    // considered compute operations. This distinguishment is needed to balance
    // CMX memory space and not to fill CMX space with only data operations resulting
    // in not being able to fit the compute operation. Data operations will only be
    // scheduled when needed by the compute operation so that the CMX space can be
    // freed as soon as possible.
    if (getExecutorType(opIdx) != config::ExecutorKind::DMA_NN) {
        return false;
    }

    if (_outputOps.find(opIdx) != _outputOps.end()) {
        return false;
    }

    // TODO: E93697 extend DataOp scheduling to schedule its dependencies
    for (const auto& depIdx : _depsInfo.getOpDeps(opIdx)) {
        if (isDataOp(depIdx)) {
            return false;
        }
    }

    if (auto dmaTask = VPUIP::getDmaTypeOp(_depsInfo.getExecuteOpAtIndex(opIdx))) {
        // DMA from DDR to NN_CMX
        auto srcMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaTask.getInput().getType()).getMemoryKind();
        auto dstMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaTask.getOutput().getType()).getMemoryKind();
        return (_memKind == dstMemSpace && _memKind != srcMemSpace);
    }

    return false;
}

// Compute operation that depends only on constants and function inputs.
bool FeasibleMemoryScheduler::isNoInputDepComputeOp(operationIdxType opIdx) {
    // #E163065 - hang to be investigated. Limit to SW dequantize only.
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (VPUIP::VPUIPDialect::getExecutorKind(execOp) != config::ExecutorKind::SHAVE_ACT) {
        return false;
    }
    auto isSWDequant = false;
    execOp.getBody()->walk([&](mlir::Operation* op) {
        if (auto swKernelOp = mlir::dyn_cast_or_null<VPUIP::SwKernelOp>(op)) {
            if (getSwKernelEntryName(swKernelOp) == "dequantize") {
                isSWDequant = true;
            }
        }
    });
    if (!isSWDequant) {
        return false;
    }

    auto opDeps = _depsInfo.getOpDeps(opIdx);
    for (auto& dep : opDeps) {
        // Depends on data op which has no other dependencies
        if (!_isDataOp[dep] || !_depsInfo.getOpDeps(dep).empty()) {
            return false;
        }
    }

    return true;
}

void FeasibleMemoryScheduler::identifyDataOps() {
    auto opCount = _inDegreeTable.size();
    _isDataOp.resize(opCount);
    for (size_t opIdx = 0; opIdx < opCount; ++opIdx) {
        if (isDataOp(opIdx)) {
            _isDataOp.set(opIdx);
        }
    }
}

bool FeasibleMemoryScheduler::freeMemoryResources(const HeapElement& hElement) {
    auto op = _depsInfo.getExecuteOpAtIndex(hElement.op_);
    // free possible buffers, where this is the last user of the buffer
    bool freeMemoryResources = false;
    for (auto& buffer : _liveRangeInfo.getUsedBuffers(op)) {
        if (_liveRangeInfo.eraseUser(buffer, op) == 0) {
            _log.nest().trace("Mark buffer as dead, '{0}'", buffer);
            _scan.handler().markAsDead(buffer);
            freeMemoryResources = true;
        }
    }
    if (freeMemoryResources) {
        _log.nest().trace("Free non alive buffers");
        _scan.freeDeadRanges();
    }
    return freeMemoryResources;
}

bool FeasibleMemoryScheduler::unscheduledOpsOnQueue(const QueueType& queueType) {
    for (auto& op : _cycleEndHeap) {
        if (op.queueType_ != queueType) {
            continue;
        }
        // queue type exists in cycle end heap
        return false;
    }
    return true;
}

void FeasibleMemoryScheduler::distributeReadyOps(llvm::ArrayRef<operationIdxType> readyOps) {
    // populate ready lists depending on op type/state
    _log.trace("Distribute new ready ops");
    _log = _log.nest();
    for (auto& readyOpIdx : readyOps) {
        if (_isDataOp[readyOpIdx]) {
            VPUX_THROW_UNLESS(_readyDataOps.find(readyOpIdx) == _readyDataOps.end(),
                              "Operation already in the ready data list '{0}'", readyOpIdx);
            _log.nest().trace("Add to ready data ops '{0}'", readyOpIdx);
            _readyDataOps.insert(readyOpIdx);
            const auto newReadyOps = reduceInDegreeOfAdjacentOperations(readyOpIdx);
            distributeReadyOps(newReadyOps);
        } else {
            const auto queueType = getQueueType(readyOpIdx);
            if (VPUIP::VPUIPDialect::isComputeExecutorKind(queueType.execKind)) {
                VPUX_THROW_UNLESS(_readyComputeOps.find(readyOpIdx) == _readyComputeOps.end(),
                                  "Operation already in ready compute list '{0}'", readyOpIdx);
                _log.nest().trace("Add to ready compute ops '{0}'", readyOpIdx);
                _readyComputeOps.insert(readyOpIdx);
            } else {
                VPUX_THROW_UNLESS(_readyDMAOps.find(readyOpIdx) == _readyDMAOps.end(),
                                  "Operation already in ready compute DMA list '{0}'", readyOpIdx);
                _log.nest().trace("Add to ready DMA ops '{0}'", readyOpIdx);
                _readyDMAOps.insert(readyOpIdx);
            }
        }
    }
    _log = _log.unnest();
}

SmallVector<operationIdxType> FeasibleMemoryScheduler::unlockNewReadyOps(const HeapElement& hElement) {
    if (!hElement.isOriginalOp()) {
        return SmallVector<operationIdxType>{};
    }
    const auto executorType = getExecutorType(hElement.op_);
    if (!VPUIP::VPUIPDialect::isComputeExecutorKind(executorType)) {
        // non compute executor kind consumers unlocked during scheduling
        return SmallVector<operationIdxType>{};
    }
    // propagate through original compute ops, generate new ready ops
    return reduceInDegreeOfAdjacentOperations(hElement.op_);
}

void FeasibleMemoryScheduler::unscheduleAllCompletingOps() {
    // find earliest scheduled queue cycle end
    const auto minScheduledQueueCycle = findMinScheduledQueueCycle();

    // unschedule operations from cycle end heap to target cycle end
    SmallVector<operationIdxType> readyOps = {};
    for (auto& nextOp : llvm::make_early_inc_range(_cycleEndHeap)) {
        if (nextOp.cycleEnd_ > minScheduledQueueCycle) {
            // do not unschedule post target cycle
            break;
        }

        _log.nest(2).trace("Unschedule opIdx '{0}'", nextOp.op_);
        if (freeMemoryResources(nextOp)) {
            // align executors only if memory resources freed
            alignExecutors(nextOp.cycleEnd_);
        }

        // retrieve new ready ops
        const auto newReadyOps = unlockNewReadyOps(nextOp);
        readyOps.insert(readyOps.end(), newReadyOps.begin(), newReadyOps.end());

        // remove op from heap
        _cycleEndHeap.erase(nextOp);
    }

    // distribute ready ops into ready lists
    distributeReadyOps(readyOps);
}

SmallVector<operationIdxType> FeasibleMemoryScheduler::reduceInDegreeOfAdjacentOperations(operationIdxType opIdx) {
    SmallVector<operationIdxType> zeroInDegreeOps;
    // reduce in-degree (number of incoming edges) for consumers of ready data ops
    for (const auto& consumer : _depsInfo.getConsumerOps(opIdx)) {
        if (_inDegreeTable[consumer] < 2) {
            zeroInDegreeOps.push_back(consumer);
            _inDegreeTable.erase(consumer);
        } else {
            VPUX_THROW_UNLESS(_inDegreeTable[consumer] > 0, "Invalid in-degree");
            _inDegreeTable[consumer]--;
        }
    }
    return zeroInDegreeOps;
}

void FeasibleMemoryScheduler::initializeReadyLists() {
    // populate ready lists with operations without dependencies
    SmallVector<operationIdxType> operationsWithNoDependencies;

    for (auto& entry : _inDegreeTable) {
        if (entry.second == 0) {
            operationsWithNoDependencies.push_back(entry.first);
        }
    }

    distributeReadyOps(operationsWithNoDependencies);
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::sortUsedBuffers(mlir::DenseSet<mlir::Value>& operationBuffers) {
    // retrieve size of buffers
    SmallVector<BufferOrder> bufferVector;
    // order buffers based on usage type
    for (auto& val : operationBuffers) {
        auto opSize = _scan.handler().getSize(val);

        size_t outDegree = 0;
        if (_bufferOpIdxMap.find(val) != _bufferOpIdxMap.end()) {
            for (auto& opIdx : _bufferOpIdxMap[val]) {
                outDegree += _outDegreeTable[opIdx];
            }
        } else {
            VPUX_THROW("Couldn't find the buffer '{0}' in output async index map", val.getLoc());
        }

        bufferVector.push_back(BufferOrder(val, opSize, outDegree));
    }
    // sort based on buffer qualities
    llvm::sort(bufferVector.begin(), bufferVector.end(), [](const BufferOrder& val1, const BufferOrder& val2) {
        // second outDegree of the buffer/parentOp
        if (val1.outDegree != val2.outDegree) {
            return val1.outDegree > val2.outDegree;
        }

        // third op size
        if (val1.size != val2.size) {
            return val1.size > val2.size;
        }

        // finally position in IR
        const auto parentOp = val1.buffer.getDefiningOp();
        VPUX_THROW_UNLESS(parentOp != nullptr, "Block arguments are not supported");
        return parentOp->isBeforeInBlock(val2.buffer.getDefiningOp());
    });

    // repopulate only with buffers
    SmallVector<mlir::Value> orderedBufs;
    for (auto& buff : bufferVector) {
        orderedBufs.push_back(buff.buffer);
    }
    return orderedBufs;
}

size_t FeasibleMemoryScheduler::scheduleSpilledOpBuffer(operationIdxType opIdx, mlir::Value* buffer) {
    // schedule the spilled dependency
    const auto queueAndCycle = getCurrentCycleAndExecutorInstanceMaskForSpill(*buffer, EOpType::IMPLICIT_SPILL_READ_OP);
    _log.nest().trace("Scheduling spilled op:'{0}' at cycle {1}", opIdx, queueAndCycle.cycle);
    // also store the buffer spilled
    auto spilledReadBuffer = *buffer;
    VPUX_THROW_UNLESS(_readySpilledOps.find(spilledReadBuffer) != _readySpilledOps.end(),
                      "Failed to find spill buffer");
    _readySpilledOps.erase(spilledReadBuffer);
    VPUX_THROW_UNLESS(_spillBufferMap.find(opIdx) != _spillBufferMap.end(), "Failed to find spill opIdx");
    if (_spillBufferMap[opIdx].size() > 1) {
        _spillBufferMap[opIdx].erase(spilledReadBuffer);
    } else {
        _spillBufferMap.erase(opIdx);
    }
    // update representation in scan handler
    _scan.handler().removeDynamicSpill(spilledReadBuffer);
    const auto opCycleCost = spilledOperationCycleCost(spilledReadBuffer);
    const auto nextAvailableCycle = queueAndCycle.cycle + opCycleCost;
    // update current cycle directly
    updateCurrentCycleForExecutor(queueAndCycle.queueType, queueAndCycle.execMask, nextAvailableCycle);
    pushToCycleBeginHeap(
            HeapElement(opIdx, queueAndCycle, opCycleCost, EOpType::IMPLICIT_SPILL_READ_OP, spilledReadBuffer));
    return nextAvailableCycle;
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::getNonAliveBuffersUsedByOperation(operationIdxType opIdx) {
    // retrieve all buffers used by the op which are not alive
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto usedBuffs = _liveRangeInfo.getUsedBuffers(op);
    SmallVector<mlir::Value> operationBuffers;

    for (auto& buffer : usedBuffs) {
        if (_scan.handler().isAlive(buffer)) {
            continue;
        }
        operationBuffers.push_back(buffer);
    }
    return operationBuffers;
}

mlir::DenseSet<mlir::Value> FeasibleMemoryScheduler::getBuffersToAllocateForOp(operationIdxType opIdx) {
    // retrieve non alive buffers
    auto usedBuffers = getNonAliveBuffersUsedByOperation(opIdx);

    mlir::DenseSet<mlir::Value> buffersToAllocate(usedBuffers.begin(), usedBuffers.end());
    for (const auto& dep : _depsInfo.getOpDeps(opIdx)) {
        if (_opIdxEndCycleMap.find(dep) != _opIdxEndCycleMap.end()) {
            // op was scheduled
            continue;
        }

        VPUX_THROW_UNLESS(_readyDataOps.find(dep) != _readyDataOps.end(),
                          "Failed to get buffers - operation not ready '{0}'", dep);
        auto depBuffers = getBuffersToAllocateForOp(dep);
        buffersToAllocate.insert(depBuffers.begin(), depBuffers.end());
    }

    return buffersToAllocate;
}

size_t FeasibleMemoryScheduler::scheduleDependencies(operationIdxType opIdx) {
    // retrieve operation's buffers that need allocation
    for (auto val : getNonAliveBuffersUsedByOperation(opIdx)) {
        _scan.handler().markAsAlive(val);
        _log.nest().trace("Mark as alive '{0}'", val);
        if (!_scan.handler().isDynamicSpill(val)) {
            continue;
        }
        // special case for spilled reads
        auto bufferProducer = _bufferProducer.find(val);
        VPUX_THROW_UNLESS(bufferProducer != _bufferProducer.end(), "Failed to find buffer producer for '{0}'", val);
        scheduleSpilledOpBuffer(bufferProducer->second, &val);
    }

    // schedule required dependencies order based on earliest scheduling cycle and IR order
    std::map<size_t, std::set<operationIdxType>> sortedDemandList;
    for (const auto& depIdx : _depsInfo.getOpDeps(opIdx)) {
        if (_opIdxEndCycleMap.find(depIdx) != _opIdxEndCycleMap.end()) {
            // op was scheduled
            continue;
        }

        VPUX_THROW_UNLESS(_readyDataOps.find(depIdx) != _readyDataOps.end(),
                          "Failed to schedule dependencies - operation not ready '{0}'", depIdx);
        const auto cycleBegin = getCurrentCycleAndExecutorInstanceMask(depIdx).cycle;
        sortedDemandList[cycleBegin].insert(depIdx);
    }

    for (auto& entry : sortedDemandList) {
        for (auto& depIdx : entry.second) {
            scheduleOp(depIdx);
            _readyDataOps.erase(depIdx);
        }
    }

    return getEarliestComputeBeginCycle(opIdx);
}

size_t FeasibleMemoryScheduler::scheduleOp(operationIdxType opIdx, EOpType opType) {
    // schedule dependencies
    const auto depEndCycle = scheduleDependencies(opIdx);

    _log.trace("Scheduling op: '{0}'", opIdx);

    // find schedule cycles for op
    const auto queueAndCycle = getCurrentCycleAndExecutorInstanceMask(opIdx, depEndCycle);
    const auto opCycleCost = operationCycleCost(opIdx);
    const auto nextAvailableCycle = queueAndCycle.cycle + opCycleCost;

    // schedule op
    updateCurrentCycleForExecutor(queueAndCycle.queueType, queueAndCycle.execMask, nextAvailableCycle);
    if (_loopPrefetchInd.count(opIdx)) {
        opType = EOpType::LOOP_OP;
    }
    pushToCycleBeginHeap(HeapElement(opIdx, queueAndCycle, opCycleCost, opType));

    return nextAvailableCycle;
}

size_t FeasibleMemoryScheduler::getOperationLevel(operationIdxType opIdx, bool isSpilled) const {
    if (!isSpilled) {
        return _opLevelVec[opIdx];
    }
    // original consumer(s) could have been already scheduled
    auto minRemainingConsumerLevel = std::numeric_limits<size_t>::max();
    for (const auto& consumerIdx : _depsInfo.getConsumerOps(opIdx)) {
        if (_opIdxEndCycleMap.find(consumerIdx) != _opIdxEndCycleMap.end()) {
            // consumer scheduled
            continue;
        }
        minRemainingConsumerLevel = std::min(minRemainingConsumerLevel, _opLevelVec[consumerIdx]);
    }
    return minRemainingConsumerLevel;
}

size_t calculateTotalBuffersSize(const mlir::DenseSet<mlir::Value>& buffers,
                                 LinearScan<mlir::Value, LinearScanHandler>& scan) {
    size_t totalSize = 0;
    auto& handler = scan.handler();
    for (const auto& buffer : buffers) {
        totalSize += handler.getSize(buffer);
    }
    return totalSize;
}

// Check if the total size of the buffers fits into the available free CMX memory
bool fitsIntoFreeCMX(LinearScan<mlir::Value, LinearScanHandler>& scan, const mlir::DenseSet<mlir::Value>& buffers,
                     size_t freeCmx) {
    size_t totalSize = calculateTotalBuffersSize(buffers, scan);
    return totalSize <= freeCmx;
}

// Extract local (non-shared) buffers used by an operation
SmallVector<mlir::Value> getLocalUsedBuffers(const OpAllocationInfo& allocInfo, const ValueOrderedSet& sharedBuffers) {
    SmallVector<mlir::Value> usedBuffers;
    for (const auto& buf : allocInfo.inBuffers) {
        if (!sharedBuffers.count(buf)) {
            usedBuffers.push_back(buf);
        }
    }
    for (const auto& buf : allocInfo.outBuffers) {
        if (!sharedBuffers.count(buf)) {
            usedBuffers.push_back(buf);
        }
    }
    return usedBuffers;
}

// Get CMX demand of operation including all its dependencies
size_t FeasibleMemoryScheduler::getOpCmxDemand(operationIdxType opIdx) {
    auto buffers = getBuffersToAllocateForOp(opIdx);
    return calculateTotalBuffersSize(buffers, _scan);
}

/// Get candidate DATA_IN operations that can be prefetched ahead of compute operations.
/// It uses a level-based filtering mechanism to prevent prefetching operations that are
/// too far ahead in the dependency graph
///
/// @param lastScheduledOp The operation index of the last scheduled compute operation.
///                        Used as an upper bound - only operations with indices <= this
///                        value are considered (IR ordering constraint).
///
/// @param lastScheduledLevel The scheduling level of the last scheduled compute operation.
///                           Levels are assigned during initialization based on compute op order.
///                           Each compute op and its dependencies get assigned the same level,
///                           and levels increment for each successive compute op in the IR.
///                           This parameter defines the "current position" in the scheduling.
///
/// @return A map of [operationLevel -> set of operation indices], sorted by level.
///         Operations at lower levels (closer to current scheduling position) are
///         prioritized for prefetching.
///
/// Example:
///       Scheduled Ops: 1      2       3       5       7        10
///                                                           [lastScheduledOp]
///       Schedule Level: 0      0       1       1       2        3
///                                                           [lastScheduledLevel]
/// Prefetch candidates can only be chosen from op id < 10 and level <= 3 + prefetchingLevelLimit
std::map<size_t, std::set<operationIdxType>> FeasibleMemoryScheduler::getPrefetchCandidates(size_t lastScheduledOp,
                                                                                            size_t lastScheduledLevel) {
    // Calculate the current level limit: operations beyond this level are too far ahead
    // to prefetch. The limit prevents potential dynamic spilling caused by too early prefetch
    // _prefetchingLevelLimit defines the prefetching "lookahead window"
    const auto currLevelLimit = lastScheduledLevel + _prefetchingLevelLimit;

    // Collect ready data operations (DMAs) that haven't been scheduled yet
    // Sort them by level - lower levels are prefetched first (closer dependencies)
    std::map<size_t, std::set<operationIdxType>> sortedCandidates;

    // Check normal data operations (non-spilled DMAs)
    for (auto& dataOp : _readyDataOps) {
        // Skip operations belonging to loop regions
        // Loop operations are handled separately by scheduleLoopRegions()
        if (_loopRegionInd.count(dataOp)) {
            continue;
        }

        // Respect IR ordering - only prefetch ops that appear before the last scheduled compute op in the IR
        if (dataOp > lastScheduledOp) {
            continue;
        }

        // Check if operation is within the prefetching window
        // Operations beyond currLevelLimit are too far in the future
        const auto opLevel = getOperationLevel(dataOp);
        if (opLevel > currLevelLimit) {
            continue;
        }

        _log.nest().trace("Prefetch candidate: '{0}'", dataOp);
        sortedCandidates[opLevel].insert(dataOp);
    }

    // Also consider spilled buffer read operations as prefetch candidates
    // These are operations whose buffers were previously evicted to DDR and need
    // to be brought back into CMX
    for (auto& spillOp : _readySpilledOps) {
        // Skip loop region operations
        if (_loopRegionInd.count(spillOp.second)) {
            continue;
        }

        // Respect IR ordering
        if (spillOp.second > lastScheduledOp) {
            continue;
        }

        // Check level limit
        // For spilled ops, use isSpilled=true to get the minimum level of
        // remaining (unscheduled) consumers
        const auto opLevel = getOperationLevel(spillOp.second, true);
        if (opLevel > currLevelLimit) {
            continue;
        }

        // Skip if the buffer is already alive (already in CMX)
        if (_scan.handler().isAlive(spillOp.first)) {
            continue;
        }

        _log.nest().trace("Prefetch spill candidate: '{0}'", spillOp.second);
        sortedCandidates[opLevel].insert(spillOp.second);
    }

    return sortedCandidates;
}

std::pair<size_t, mlir::DenseSet<mlir::Value>> FeasibleMemoryScheduler::getPrefetchInfo(operationIdxType opIdx) {
    mlir::DenseSet<mlir::Value> operationBuffers;
    size_t scheduleCycle = 0;
    if (_readyDataOps.find(opIdx) != _readyDataOps.end()) {
        operationBuffers = getBuffersToAllocateForOp(opIdx);
        scheduleCycle = getCurrentCycleAndExecutorInstanceMask(opIdx).cycle;
    } else {
        if (_spillBufferMap.find(opIdx) == _spillBufferMap.end()) {
            bool wasScheduledBefore =
                    std::any_of(_cycleBeginHeap.begin(), _cycleBeginHeap.end(), [=](const HeapElement& elem) {
                        return elem.op_ == opIdx;
                    });
            VPUX_THROW_UNLESS(wasScheduledBefore, "Failed to find spill candidate '{0}'", opIdx);
            _log.nest(2).trace("Prefetch '{0}' has been already scheduled", opIdx);
        } else {
            operationBuffers = _spillBufferMap[opIdx];
            for (auto& val : operationBuffers) {
                const auto queueAndCycle =
                        getCurrentCycleAndExecutorInstanceMaskForSpill(val, EOpType::IMPLICIT_SPILL_READ_OP);
                scheduleCycle = std::max(scheduleCycle, queueAndCycle.cycle);
            }
        }
    }
    return {scheduleCycle, operationBuffers};
}

/// Execute prefetching for a single operation by scheduling it ahead of its consumer.
/// @param opIdx Operation index to prefetch
/// @return Number of operations scheduled (1 for normal ops, N for spilled ops with multiple buffers)
size_t FeasibleMemoryScheduler::executePrefetch(operationIdxType opIdx) {
    if (_readyDataOps.find(opIdx) != _readyDataOps.end()) {
        // Normal data operation (e.g., DMA from DDR to CMX)
        _log.nest().trace("Scheduling prefetch op: '{0}'", opIdx);
        scheduleOp(opIdx, EOpType::ORIGINAL_PREFETCHED_OP);
        _readyDataOps.erase(opIdx);
        return 1UL;
    } else {
        // Spilled buffer read operation (buffer was evicted to DDR, bring it back)
        auto spilledBuffers = _spillBufferMap[opIdx];
        auto spilledBuffersVec = sortUsedBuffers(spilledBuffers);
        for (auto& val : spilledBuffersVec) {
            _log.nest().trace("Scheduling spill prefetch op: '{0}'", opIdx);
            // mark the spilled buffer as alive in case other operation
            // that can be scheduled as part of this prefetching iteration also depends on it
            _scan.handler().markAsAlive(val);
            _log.nest().trace("Mark as alive '{0}'", val);
            scheduleSpilledOpBuffer(opIdx, &val);
        }
        // Return number of spill-read DMAs scheduled
        return spilledBuffersVec.size();
    }
}

void FeasibleMemoryScheduler::prefetchOps(ArrayRef<std::pair<operationIdxType, size_t>> scheduledOps,
                                          mlir::DenseSet<mlir::Value>& buffersToAllocate,
                                          bool checkMemoryFragmentation) {
    // consider barrier limitations
    std::set<operationIdxType> aliveOperations;
    for (auto& aliveBuffer : _scan.handler().getAliveValues()) {
        const auto aliveOpIdx = _bufferProducer[aliveBuffer];
        if (!_isDataOp[aliveOpIdx]) {
            continue;
        }
        aliveOperations.insert(aliveOpIdx);
    }

    // TODO: E93149 update barrier usage
    auto aliveOperationCount = aliveOperations.size();
    size_t barrierLimit = checked_cast<size_t>(_barrierPerCluster * _nceClusterCount) - scheduledOps.size();

    if (barrierLimit <= aliveOperationCount) {
        _log.nest().trace("Can not prefetch: alive ops '{0}' >= barrier limit '{1}'", aliveOperationCount,
                          barrierLimit);
        return;
    }

    // use IR order for prefetching
    size_t lastScheduledOp = 0;
    size_t lastScheduledCycle = 0;
    size_t lastScheduledLevel = 0;
    for (auto& scheduledOp : scheduledOps) {
        const auto executorType = getExecutorType(scheduledOp.first);
        if (executorType == config::ExecutorKind::DMA_NN) {
            continue;
        }
        lastScheduledOp = std::max(lastScheduledOp, scheduledOp.first);
        lastScheduledLevel = std::max(lastScheduledLevel, _opLevelVec[scheduledOp.first]);
        if (operationCycleCost(scheduledOp.first) <= 1) {
            // avoid comparing invalid cycles
            continue;
        }
        lastScheduledCycle = std::max(lastScheduledCycle, scheduledOp.second);
    }

    auto sortedCandidates = getPrefetchCandidates(lastScheduledOp, lastScheduledLevel);

    // try to allocate and schedule prefetch ops
    for (auto& entry : sortedCandidates) {
        for (const auto& opIdx : entry.second) {
            // get info about prefetch
            auto [scheduleCycle, operationBuffers] = getPrefetchInfo(opIdx);

            if (operationBuffers.empty()) {
                _log.nest(2).trace("No buffers to allocate for: '{0}'", opIdx);
                continue;
            }

            // avoid barriers for next compute dependencies using level check
            if (lastScheduledCycle != 0 && lastScheduledLevel + 1 < _opLevelVec[opIdx] &&
                scheduleCycle >= lastScheduledCycle) {
                _log.nest(2).trace("Would be scheduled after compute: '{0}'", opIdx);
                return;
            }

            operationBuffers.insert(buffersToAllocate.begin(), buffersToAllocate.end());
            if (!canAllocBuffers(operationBuffers)) {
                _log.nest(2).trace("Can not fit: '{0}'", opIdx);

                if (checkMemoryFragmentation) {
                    // save the failed prefetch buffer set for later allocation
                    // will try to resolve the fragment by actively spilling
                    if (fitsIntoFreeCMX(_scan, operationBuffers, _scan.totalFreeSize())) {
                        _log.nest(2).trace("Prefetch fails due to memory fragmentation");
                        _prefetchFailedDueToFragmentation = true;
                        ++_prefetchFragmentationFailureCount;
                        _fragmentedBuffers = std::move(operationBuffers);
                    }
                }

                return;
            }

            // need to allocate more buffers
            buffersToAllocate = std::move(operationBuffers);

            // schedule prefetch op
            aliveOperationCount += executePrefetch(opIdx);

            // consider barrier limitations
            if (barrierLimit <= aliveOperationCount) {
                _log.nest().trace("End prefetch: alive ops '{0}' >= barrier limit '{1}'", aliveOperationCount,
                                  barrierLimit);
                return;
            }
        }
    }
}

void FeasibleMemoryScheduler::resetScheduleComputeOpsDeltas() {
    _prefetchFailedDueToFragmentation = false;
    _originalBufferProducersFromScheduleComputeOps.clear();
    _newBufferProducersFromScheduleComputeOps.clear();
    _originalOpIdxEndCycleMapFromScheduleComputeOps.clear();
    _newOpsFromScheduleComputeOps.clear();
    _originalBufferLastCycleUseFromScheduleComputeOps.clear();
    _newBufferLastCycleUsesFromScheduleComputeOps.clear();
}

std::vector<size_t> FeasibleMemoryScheduler::getReadyLoopOps() {
    // Keep the compute op order
    // Record the first unscheduled operation in each compute queue
    // This is used to preserve IR ordering: loop operations should not be scheduled
    // before non-loop operations that appear earlier in the IR for the same queue.
    //
    // Why this matters:
    // - _computeOpOrder maintains operations in their original IR order per queue
    // - If a loop contains NCE op at index 100, but there's a non-loop NCE op at index 50
    //   that hasn't been scheduled yet, we must schedule the non-loop op first
    // - This prevents reordering that could violate dependencies or compiler assumptions
    std::map<FeasibleMemoryScheduler::QueueType, operationIdxType> nonLoopNextIdx;
    for (auto& queue : _computeOpOrder) {
        auto firstOpInQueue = queue.second.begin();
        if (firstOpInQueue == queue.second.end()) {
            continue;
        }
        nonLoopNextIdx[queue.first] = *firstOpInQueue;
    }

    std::vector<size_t> readyLoopOps;
    for (size_t idx = 0; idx < _loopRegions.size(); idx++) {
        const auto& computeRegion = _loopRegions[idx];
        if (computeRegion.getLoopType() == LoopType::None) {
            continue;
        }
        if (_scheduledLoopRegionInd.count(idx)) {
            continue;
        }

        _log.trace("Loop region {0}", idx);

        bool loopRegionReady = true;
        for (auto& opIdx : computeRegion.dependencies) {
            if (_opIdxEndCycleMap.count(opIdx) || _readyDataOps.count(opIdx)) {
                continue;
            }
            _log.nest().trace("Global dep not ready {0}", opIdx);
            loopRegionReady = false;
            break;
        }

        // Collect all 1st loop operation indices
        mlir::DenseSet<size_t> loopOps;
        for (const auto& opInfo : computeRegion.schedulingLoop->loopBodies[0]) {
            loopOps.insert(opInfo.opIdx);
        }

        // Check deps for 1st loop
        for (const auto& opInfo : computeRegion.schedulingLoop->loopBodies[0]) {
            // some deps can be added for compute op ordering
            for (const auto& depIdx : _depsInfo.getOpDeps(opInfo.opIdx)) {
                if (loopOps.count(depIdx) || _opIdxEndCycleMap.count(depIdx) || _readyDataOps.count(depIdx)) {
                    continue;
                }
                _log.nest().trace("Not ready {0} for {1}", depIdx, opInfo.opIdx);
                loopRegionReady = false;
                break;
            }
            if (opInfo.allocationType == AllocationType::COMPUTE) {
                const auto queueType = getQueueType(opInfo.opIdx);
                if (nonLoopNextIdx.count(queueType) && nonLoopNextIdx[queueType] < opInfo.opIdx) {
                    loopRegionReady = false;
                }
            }
            if (!loopRegionReady) {
                break;
            }
        }

        if (!loopRegionReady) {
            break;
        }
        readyLoopOps.push_back(idx);
    }
    return readyLoopOps;
}

// Clean up schedule status by unscheduling operations and collecting ready ops
// This prepares the scheduler state for loop scheduling
void FeasibleMemoryScheduler::cleanupBeforeLoopScheduling(SmallVector<operationIdxType>& readyOps) {
    // Move all operations from cycle begin to cycle end heap
    moveFromCycleBeginToCycleEndHeap();

    // Unschedule all operations in the cycle end heap
    // This may include spilled buffers that need to be deallocated
    for (auto& nextOp : llvm::make_early_inc_range(_cycleEndHeap)) {
        _log.nest(2).trace("Unschedule opIdx '{0}'", nextOp.op_);

        // Free memory resources if possible
        if (freeMemoryResources(nextOp)) {
            // Align executors only if memory resources were freed
            alignExecutors(nextOp.cycleEnd_);
        }

        // Collect new ready operations that become available after unscheduling
        const auto newReadyOps = unlockNewReadyOps(nextOp);
        readyOps.insert(readyOps.end(), newReadyOps.begin(), newReadyOps.end());

        // Remove operation from heap
        _cycleEndHeap.erase(nextOp);
    }
}

// Schedule dependencies that must execute before the loop starts
// These are operations that produce inputs required by the loop
void FeasibleMemoryScheduler::schedulePreLoopDependencies(const ComputeRegion& computeRegion) {
    for (auto& opIdx : computeRegion.dependencies) {
        // Skip if already scheduled
        if (_opIdxEndCycleMap.count(opIdx)) {
            _log.nest(2).trace("Already scheduled dependency {0}", opIdx);
            continue;
        }

        // Schedule the dependency operation
        _log.nest().trace("Schedule dependency {0}", opIdx);
        scheduleOp(opIdx);

        // Remove from ready data ops if present
        if (_readyDataOps.count(opIdx)) {
            _readyDataOps.erase(opIdx);
        }
    }
}

// Schedule spill operations for shared buffers that are dynamically spilled
// Shared buffers are marked as alive and spill-read operations are scheduled
void FeasibleMemoryScheduler::scheduleLoopSpills(const ComputeRegion& computeRegion) {
    for (auto val : computeRegion.sharedExternalBuffers) {
        // Mark buffer as alive for the duration of the loop
        // If the buffer is already alive, nothing is changed
        _scan.handler().markAsAlive(val);
        _log.nest().trace("Mark as alive '{0}'", val);

        // Check if buffer is dynamically spilled
        if (!_scan.handler().isDynamicSpill(val)) {
            continue;
        }

        // Schedule spill-read operation for this buffer
        auto bufferProducer = _bufferProducer.find(val);
        VPUX_THROW_UNLESS(bufferProducer != _bufferProducer.end(), "Failed to find buffer producer for '{0}'", val);
        _log.nest(2).trace("scheduleSpilledOpBuffer {0}", bufferProducer->second);
        scheduleSpilledOpBuffer(bufferProducer->second, &val);
    }
}

// Verify buffer usage across all loop iterations and collect information about:
// - Dynamic spill buffers: buffers that are spilled and need special handling
// - Prefetch buffers: buffers that can be prefetched before the loop starts
void FeasibleMemoryScheduler::verifyAndCollectLoopBuffers(
        const ComputeRegion& computeRegion, mlir::DenseMap<size_t, mlir::DenseSet<mlir::Value>>& dynamicSpillBuffers,
        mlir::DenseMap<mlir::Value, vpux::AddressType>& prefetchBuffers) {
    const auto& sharedBuffers = computeRegion.sharedExternalBuffers;
    // Verify each loop iteration
    for (size_t i = 0; i < computeRegion.schedulingLoop->loopBodies.size(); i++) {
        const auto& loop = computeRegion.schedulingLoop->loopBodies[i];
        _log.nest().trace("Verify ops from loop {0}", i + 1);

        for (auto& allocInfo : loop) {
            const auto opIdx = allocInfo.opIdx;
            _log.nest(2).trace("Verify loop op {0}", opIdx);

            const auto usedLocalBuffers = getLocalUsedBuffers(allocInfo, sharedBuffers);

            if (_opIdxEndCycleMap.count(opIdx)) {
                // Operation was pre-scheduled (before loop scheduling started)
                // Two situations can lead to pre-scheduling:
                // 1. The operation is prefetched (not spilling). Save it into prefetchBuffers for later address
                // assignment
                // 2. The operation is prefetched but spilled.
                _log.nest(3).trace("Pre-scheduled op {0}", opIdx);

                // Verify all pre-scheduled ops are either spilled or prefetchable
                for (auto buffer : usedLocalBuffers) {
                    if (!_scan.handler().isDynamicSpill(buffer)) {
                        // Case 1: DATA_IN operations can be prefetched (and not spilled)
                        if (allocInfo.allocationType == vpux::AllocationType::DATA_IN) {
                            VPUX_THROW_UNLESS(usedLocalBuffers.size() == 1,
                                              "Prefetch op should have only 1 buffer, got {0}",
                                              usedLocalBuffers.size());
                            // If the operation is scheduled and not spilled which means this buffer is still in CMX
                            // then directly record its address
                            prefetchBuffers[buffer] = _scan.handler().getAddress(buffer);
                            continue;
                        }
                        cleanUpAndLogSchedule(_scheduledOps);
                        VPUX_THROW("Loop op should be spilled {0}", opIdx);
                    } else {
                        // Case 2: prefetched but spilled
                        _log.nest(4).trace("Spilled buffer from loop op {0}", opIdx);
                    }
                }
            } else {
                // Operation not yet scheduled - collect dynamically spilled buffers
                for (auto& buffer : usedLocalBuffers) {
                    if (_scan.handler().isDynamicSpill(buffer)) {
                        dynamicSpillBuffers[opIdx].insert(buffer);
                    }
                }
            }
        }
    }

    // Free buffers that are no longer alive
    _scan.freeDeadRanges();
}

// Unschedule operations up to a target cycle
// This helper iterates through the cycle end heap and removes operations
// that complete at or before the target cycle, freeing their memory resources
void FeasibleMemoryScheduler::unscheduleOpsToCycle(size_t targetUnscheduleCycle) {
    for (auto& nextOp : llvm::make_early_inc_range(_cycleEndHeap)) {
        if (nextOp.cycleEnd_ > targetUnscheduleCycle) {
            // Do not unschedule operations past target cycle
            break;
        }

        _log.nest(2).trace("Unschedule opIdx '{0}'", nextOp.op_);
        if (freeMemoryResources(nextOp)) {
            // Align executors only if memory resources were freed
            alignExecutors(nextOp.cycleEnd_);
        }

        // Remove operation from heap
        _cycleEndHeap.erase(nextOp);
    }
}

// Unschedule operations from a specific loop iteration
// This helper unschedules operations up to the loop's end cycle and spills
// all local buffers that are still alive to ensure correct loop iteration overlap
void FeasibleMemoryScheduler::unscheduleLoopOps(size_t loopIdx, size_t targetUnscheduleCycle,
                                                const SmallVector<mlir::Value>& loopBuffersForIndex) {
    // Unschedule operations to target cycle
    unscheduleOpsToCycle(targetUnscheduleCycle);

    // Spill all local alive buffers from the loop
    const auto aliveValues = _scan.handler().getAliveValues();
    mlir::DenseSet<mlir::Value> processedBuffers;
    for (const auto& buffer : loopBuffersForIndex) {
        if (!aliveValues.count(buffer) || processedBuffers.count(buffer)) {
            continue;
        }
        processedBuffers.insert(buffer);

        // Memory will be re-used, need to spill for all future uses
        // This should be optimized in DMA spill optimization
        VPUX_THROW_UNLESS(_bufferProducer.find(buffer) != _bufferProducer.end(),
                          "Buffer not scheduled yet, invalid eviction candidate");
        auto executeOpIdx = _bufferProducer[buffer];
        // In special case of multiple output buffers, store output index
        auto outputIdx = getOpBufferOutputIdx(executeOpIdx, buffer);
        auto evictionCandidate = EvictionCandidate(/*priority=*/0UL, /*earliestConsumerIdx=*/0UL, /*size=*/1,
                                                   executeOpIdx, outputIdx, buffer);

        _log.nest(2).trace("performEvictionAndScheduling '{0}'", executeOpIdx);
        // Ensure prefetch functional for next loop iteration, do not align executors here
        auto cycleAfterSpill = performEvictionAndScheduling(evictionCandidate, false);
        moveFromCycleBeginToCycleEndHeap();
        _evictionCandidatesCache.clear();

        // For overlapped loops, cycles are aligned here
        targetUnscheduleCycle = std::max(targetUnscheduleCycle, cycleAfterSpill);
    }

    _log.nest().trace("Unschedule loopIdx {0} align cycles to {1}", loopIdx, targetUnscheduleCycle);
    for (auto& [queue, instances] : _executorPipelines) {
        for (auto& instance : instances) {
            instance = std::max(instance, targetUnscheduleCycle);
        }
    }

    // Unschedule operations from cycle end heap to target cycle end
    unscheduleOpsToCycle(targetUnscheduleCycle);
}

// Schedule all iterations of the loop with double buffering
// This is the core loop scheduling logic that handles:
// - Double buffering to overlap loop iterations
// - Full pipelining (unschedules loop-2 when scheduling current loop)
// - Buffer address assignment alternating between two buffer sets
void FeasibleMemoryScheduler::scheduleLoopIterationAtIndex(
        const ComputeRegion& computeRegion, size_t loopIndex, vpux::AddressType reserveOffset,
        mlir::DenseMap<size_t, size_t>& loopCycleEnd, mlir::DenseMap<size_t, SmallVector<mlir::Value>>& loopBuffers,
        const mlir::DenseMap<size_t, mlir::DenseSet<mlir::Value>>& dynamicSpillBuffers,
        const mlir::DenseMap<mlir::Value, vpux::AddressType>& prefetchBuffers, size_t& loopMaxCycleEnd) {
    const auto& localAddressMap = computeRegion.bufferAddressVec.first;
    const auto& local2AddressMap = computeRegion.bufferAddressVec.second;
    const auto& sharedBuffers = computeRegion.sharedExternalBuffers;
    const auto prefetchOpCount = computeRegion.prefetchOpCount;

    size_t bufIndex = 0;
    size_t currPrefetchCount = 0;
    mlir::DenseMap<mlir::Value, vpux::AddressType> bufferAddresses;

    // Unschedule loop-2 for full pipelining
    // This allows overlap between loop iterations: loop[i], loop[i-1], and loop[i-2]
    if (loopIndex > 1) {
        auto targetUnscheduleCycle = loopCycleEnd[loopIndex - 2];
        unscheduleLoopOps(loopIndex - 2, targetUnscheduleCycle, loopBuffers[loopIndex - 2]);
    }

    // Schedule each operation in the current loop iteration
    for (const auto& allocInfo : computeRegion.schedulingLoop->loopBodies[loopIndex]) {
        // Check if can prefetch more or need to unschedule previous loop
        if (loopIndex > 0 && currPrefetchCount >= prefetchOpCount) {
            // Without full pipelining prefetching may be limited
            auto targetUnscheduleCycle = loopCycleEnd[loopIndex - 1];
            unscheduleLoopOps(loopIndex - 1, targetUnscheduleCycle, loopBuffers[loopIndex - 1]);
        }

        const auto opIdx = allocInfo.opIdx;
        _log.nest(2).trace("Schedule loop op {0}", opIdx);
        auto usedLocalBuffers = getLocalUsedBuffers(allocInfo, sharedBuffers);
        loopBuffers[loopIndex].insert(loopBuffers[loopIndex].end(), usedLocalBuffers.begin(), usedLocalBuffers.end());
        // Assign addresses to buffers (alternating between two sets for double buffering)
        for (const auto& buffer : usedLocalBuffers) {
            vpux::AddressType address = 0;
            if (bufferAddresses.find(buffer) != bufferAddresses.end()) {
                address = bufferAddresses[buffer];
            } else if (prefetchBuffers.count(buffer)) {
                address = prefetchBuffers.at(buffer);
                ++bufIndex;
            } else {
                const auto bufferOffset = (loopIndex % 2 != 0) ? local2AddressMap[bufIndex] : localAddressMap[bufIndex];
                address = reserveOffset + bufferOffset;
                ++bufIndex;
            }
            _scan.handler().setAddress(buffer, address);
            bufferAddresses[buffer] = address;
        }

        // This conditional handles two distinct execution paths:
        //   Operation was pre-scheduled (in _opIdxEndCycleMap) - handle spilled buffers
        //   Operation not yet scheduled - schedule it now
        if (_opIdxEndCycleMap.count(opIdx)) {
            // For pre-scheduled operation
            // This operation was already scheduled in an earlier phase. This can happen
            // when:
            //   1. The operation was prefetched before the loop started
            //   2. The operation was scheduled as a pre-loop dependency
            //   3. The operation was scheduled in a previous loop iteration that's
            //      still active due to pipelining
            //
            // Since the operation is already scheduled, we only need to handle any
            // buffers that were dynamically spilled to DDR to free up CMX space.
            _log.nest(2).trace("Pre-scheduled op {0}", opIdx);

            // Iterate through all buffers used by this operation
            for (auto& buffer : usedLocalBuffers) {
                // Check if this buffer was dynamically spilled (moved from CMX to DDR)
                // Dynamic spilling happens when the scheduler ran out of CMX memory and
                // had to evict some buffers to DDR to make room for new operations
                if (!_scan.handler().isDynamicSpill(buffer)) {
                    continue;  // Buffer is still in CMX and the op is scheduled, no action needed
                }

                // The buffer was spilled to DDR, so we need to read it back to CMX
                // before this operation can use it. Find the operation that produced
                // this buffer
                auto bufferProducer = _bufferProducer.find(buffer);
                VPUX_THROW_UNLESS(bufferProducer != _bufferProducer.end(), "Failed to find buffer producer for '{0}'",
                                  buffer);

                // Schedule a DMA operation to read the spilled buffer from DDR back to CMX
                // This creates an implicit spill-read operation that will be scheduled
                // before the current operation can execute
                _log.nest(3).trace("Need to read spilled buffer of op {0}", bufferProducer->second);
                scheduleSpilledOpBuffer(bufferProducer->second, &buffer);
            }
        } else {
            // The operation has not been scheduled yet, so we need to:
            //   1. Ensure all dependencies are satisfied (op is "ready")
            //   2. Schedule the operation (allocate resources, assign cycle time)
            //   3. Update tracking data structures
            //   4. Remove from ready lists

            // If the operation is not already in a ready list, we need to check if it
            // can become ready by reducing the in-degree of its dependencies
            if (!_readyDataOps.count(opIdx)) {
                auto newReadyOps = reduceInDegreeOfAdjacentOperations(opIdx);
                distributeReadyOps(newReadyOps);
            }

            auto nextAvailableCycle = scheduleOp(opIdx, EOpType::LOOP_OP);

            // Update the maximum cycle end time across all operations in the loop
            // This tracks when the entire loop iteration will complete
            loopMaxCycleEnd = std::max(loopMaxCycleEnd, nextAvailableCycle);
            loopCycleEnd[loopIndex] = loopMaxCycleEnd;

            // remove from ready ops
            if (_readyDataOps.count(opIdx)) {
                _readyDataOps.erase(opIdx);
            } else if (_readyComputeOps.count(opIdx)) {
                _readyComputeOps.erase(opIdx);
            } else if (_readyDMAOps.count(opIdx)) {
                _readyDMAOps.erase(opIdx);
            } else {
                // ERROR: Operation was not in any ready list, but we tried to schedule it
                // This should never happen - log dependencies for debugging
                for (auto dep : _depsInfo.getOpDeps(opIdx)) {
                    _log.nest().trace("Dep {0}", dep);
                }
                VPUX_THROW("Loop op not ready {0}", opIdx);
            }
        }

        if (dynamicSpillBuffers.count(opIdx)) {
            for (auto& buffer : dynamicSpillBuffers.at(opIdx)) {
                if (_scan.handler().isDynamicSpill(buffer)) {
                    VPUX_THROW("Spilled buffer should be scheduled");
                }
            }
        }

        ++currPrefetchCount;
    }
    // schedule profiling ops
    scheduleNonComputeOps();
    moveFromCycleBeginToCycleEndHeap();
}

// Handle prefetching for the last two loop iterations (boundary conditions)
// This optimizes the loop epilogue by prefetching operations after the loop
void FeasibleMemoryScheduler::handleLoopBoundaryPrefetch(
        const ComputeRegion& computeRegion, vpux::AddressType reserveOffset,
        const mlir::DenseMap<size_t, size_t>& loopCycleEnd,
        const mlir::DenseMap<size_t, SmallVector<mlir::Value>>& loopBuffers,
        const mlir::DenseMap<mlir::Value, vpux::AddressType>& prefetchBuffers) {
    // Get reserved addresses for the last two loops: loops at index loopSize-2 and loopSize-1
    mlir::DenseSet<mlir::Value> usedBufferSet;
    SmallVector<std::pair<vpux::AddressType, vpux::AddressType>> loop1Addresses;
    for (auto& buffer : loopBuffers.at(computeRegion.schedulingLoop->loopBodies.size() - 1)) {
        if (usedBufferSet.count(buffer) || prefetchBuffers.count(buffer)) {
            continue;
        }
        usedBufferSet.insert(buffer);
        const auto address = _scan.handler().getAddress(buffer);
        const auto size = _scan.handler().getSize(buffer);
        loop1Addresses.push_back({address, size});
    }
    // loop-2 uses all reserved memory
    SmallVector<std::pair<vpux::AddressType, vpux::AddressType>> loop2Addresses;
    loop2Addresses.push_back({reserveOffset, computeRegion.size});

    auto getScheduledInfo = [&](size_t loopIdx) {
        size_t lastScheduledOp = 0;
        size_t lastScheduledLevel = 0;
        size_t lastScheduledCycle = 0;
        for (const auto& allocInfo : computeRegion.schedulingLoop->loopBodies[loopIdx]) {
            lastScheduledOp = std::max(lastScheduledOp, allocInfo.opIdx);
            lastScheduledLevel = std::max(lastScheduledLevel, _opLevelVec[allocInfo.opIdx]);
            auto findPtr = _opIdxEndCycleMap.find(allocInfo.opIdx);
            VPUX_THROW_UNLESS(findPtr != _opIdxEndCycleMap.end(), "Failed to find scheduled op '{0}'", allocInfo.opIdx);
            lastScheduledCycle = std::max(lastScheduledCycle, findPtr->second);
        }
        return std::make_tuple(lastScheduledOp, lastScheduledLevel, lastScheduledCycle);
    };

    auto loopPrefetch = [&](size_t loopIdx,
                            SmallVector<std::pair<vpux::AddressType, vpux::AddressType>>& reservedRanges) {
        auto [lastScheduledOp, lastScheduledLevel, lastScheduledCycle] = getScheduledInfo(loopIdx);
        _log.nest(3).trace("lastScheduledOp {0} lastScheduledLevel {1} lastScheduledCycle {2}", lastScheduledOp,
                           lastScheduledLevel, lastScheduledCycle);
        auto sortedCandidates = getPrefetchCandidates(lastScheduledOp, lastScheduledLevel);

        mlir::DenseSet<mlir::Value> buffersToPrefetch;
        for (auto& entry : sortedCandidates) {
            for (const auto& opIdx : entry.second) {
                // get info about prefetch
                auto [scheduleCycle, operationBuffers] = getPrefetchInfo(opIdx);

                if (operationBuffers.empty()) {
                    _log.nest(2).trace("No buffers to allocate for: '{0}'", opIdx);
                    continue;
                }

                // avoid barriers for next compute dependencies using level check
                if (lastScheduledCycle != 0 && lastScheduledLevel + 1 < _opLevelVec[opIdx] &&
                    scheduleCycle >= lastScheduledCycle) {
                    _log.nest(2).trace("Would be scheduled after compute: '{0}'", opIdx);
                    return buffersToPrefetch;
                }

                operationBuffers.insert(buffersToPrefetch.begin(), buffersToPrefetch.end());
                if (!canAllocBuffersWithReservedRanges(reservedRanges, operationBuffers)) {
                    _log.nest(2).trace("Can not fit: '{0}'", opIdx);
                    return buffersToPrefetch;
                }

                // need to allocate more buffers
                buffersToPrefetch = std::move(operationBuffers);

                // schedule prefetch op
                executePrefetch(opIdx);
            }
        }

        return buffersToPrefetch;
    };

    for (size_t i = computeRegion.schedulingLoop->loopBodies.size() - 2;
         i < computeRegion.schedulingLoop->loopBodies.size(); i++) {
        // prefetch operations for loop-2 and loop-1
        _log.nest(2).trace("Try to prefetch for loop -{0}", computeRegion.schedulingLoop->loopBodies.size() - i);
        auto& reservedRanges =
                i == computeRegion.schedulingLoop->loopBodies.size() - 2 ? loop2Addresses : loop1Addresses;
        auto buffersToPrefetch = loopPrefetch(i, reservedRanges);
        // allocate prefetch ops
        sortAndAllocateBuffersWithReservedRanges(reservedRanges, buffersToPrefetch);
        // Finally unschedule loop
        auto targetUnscheduleCycle = loopCycleEnd.at(i);
        unscheduleLoopOps(i, targetUnscheduleCycle, loopBuffers.at(i));
    }
}

// Schedule loop regions with specialized ping-pong memory allocation for pipelining.
// This function orchestrates the scheduling of loop compute regions (e.g., tiled operations)
// by allocating ping-pong buffers and scheduling iterations with double-buffering support.
//
// Workflow:
// 1. Check memory requirements (shared buffers + loop working set must fit in CMX)
// 2. Clean up scheduler state by unscheduling completing operations
// 3. Schedule pre-loop global dependencies (e.g., shared input DMAs that must complete before loop starts)
// 4. Schedule spill-read operations for any shared buffers that were dynamically spilled
// 5. Allocate shared buffers and reserve contiguous space for loop local buffers
// 6. Schedule each loop iteration
// 7. Handle boundary prefetching after the last two iterations
//    to enable prefetching between the last loop (loop-1) and the previous loop than the last loop (loop-2)
// 8. Recursively schedule more ready loop regions if any were successfully scheduled
//
// Returns true if at least one loop region was scheduled, false otherwise.
bool FeasibleMemoryScheduler::scheduleLoopRegions() {
    auto readyLoopOps = getReadyLoopOps();
    bool scheduledLoop = false;

    for (auto& readyLoopOp : readyLoopOps) {
        _log.trace("Ready loop idx {0}", readyLoopOp);
        const auto& computeRegion = _loopRegions[readyLoopOp];

        // Step 1: Check if loop fits in memory (shared buffers + loop working set)
        // Only schedule the loop region when the buffers are ready to be allocated
        mlir::DenseSet<mlir::Value> buffersToAllocate;

        // Collect shared buffers that need allocation
        // Shared buffers are allocated outside the loop and kept alive across iterations
        for (auto& shared : computeRegion.sharedExternalBuffers) {
            if (_scan.handler().isAlive(shared)) {
                continue;
            }
            buffersToAllocate.insert(shared);
        }

        _log.trace("Loop step 1. Memory requirement check for loop {0}", readyLoopOp);
        _log.nest().trace("totalFreeSize {0}", _scan.totalFreeSize());
        _log.nest().trace("Need to allocate reserve {0} with alignment {1}", computeRegion.size,
                          computeRegion.baseAlignment);
        if (!canAllocBuffersWithReservedSize(buffersToAllocate, computeRegion.size, computeRegion.baseAlignment)) {
            _log.nest().trace("Failed to allocate memory for loop {0}", readyLoopOp);
            return false;
        }

        // Step 2: Clean up the schedule status and prepare for loop scheduling
        SmallVector<operationIdxType> readyOps;
        _log.trace("Loop step 2. Clean up the schedule status");
        cleanupBeforeLoopScheduling(readyOps);

        // Step 3: Schedule dependencies that must execute before the loop starts
        _log.trace("Loop step 3. Schedule pre-loop dependencies");
        schedulePreLoopDependencies(computeRegion);

        // Step 4: Schedule spill operations for shared buffers
        _log.trace("Loop step 4. Schedule spills for shared buffers");
        scheduleLoopSpills(computeRegion);

        // Step 5: Allocate memory for shared buffers and reserve space for loop working set
        _log.trace("Loop step 5. Allocate memory for shared buffers");
        const auto reserveOffset = sortAndAllocateBuffersWithReservedSize(buffersToAllocate, computeRegion.size,
                                                                          computeRegion.baseAlignment);
        _log.nest().trace("reserveOffset '{0}'", reserveOffset);

        // 6. Clean up after scheduling dependencies
        // unschedule dependency operations, propagate ready status
        // and free memory resources to prepare for loop scheduling
        _log.trace("Loop step 6. Clean up the schedule status");
        cleanupBeforeLoopScheduling(readyOps);

        // 7. Schedule loop
        _log.trace("Loop step 7. Schedule loop idx {0}", readyLoopOp);
        mlir::DenseMap<size_t, mlir::DenseSet<mlir::Value>> dynamicSpillBuffers;
        mlir::DenseMap<mlir::Value, vpux::AddressType> prefetchBuffers;
        // Verify buffer usage and collect information about dynamic spills and prefetches
        verifyAndCollectLoopBuffers(computeRegion, dynamicSpillBuffers, prefetchBuffers);

        // Get FIFO states and initialize loop scheduling
        auto maxFifoCycle = std::numeric_limits<size_t>::min();
        for (auto& [queue, instances] : _executorPipelines) {
            for (auto& instance : instances) {
                maxFifoCycle = std::max(maxFifoCycle, instance);
            }
        }
        auto loopMaxCycleEnd = maxFifoCycle;
        distributeReadyOps(readyOps);

        // Track cycle ends and buffers for each loop iteration
        mlir::DenseMap<size_t, size_t> loopCycleEnd;
        mlir::DenseMap<size_t, SmallVector<mlir::Value>> loopBuffers;

        // Schedule each loop iteration with double buffering
        for (size_t loopIndex = 0; loopIndex < computeRegion.schedulingLoop->loopBodies.size(); loopIndex++) {
            scheduleLoopIterationAtIndex(computeRegion, loopIndex, reserveOffset, loopCycleEnd, loopBuffers,
                                         dynamicSpillBuffers, prefetchBuffers, loopMaxCycleEnd);
        }

        // Step 8: Handle prefetching for the last two loop iterations (boundary conditions)
        _log.nest().trace("Loop step 8. Schedule the last two loops {0} and {1}",
                          computeRegion.schedulingLoop->loopBodies.size() - 2,
                          computeRegion.schedulingLoop->loopBodies.size() - 1);
        handleLoopBoundaryPrefetch(computeRegion, reserveOffset, loopCycleEnd, loopBuffers, prefetchBuffers);
        _log.nest().trace("data ops {0}", _readyDataOps);
        _log.nest().trace("compute ops {0}", _readyComputeOps);
        _log.nest().trace("DMA ops {0}", _readyDMAOps);

        _scheduledLoopRegionInd.insert(readyLoopOp);
        scheduledLoop = true;
    }

    if (scheduledLoop) {
        // try to schedule more loop regions
        scheduleLoopRegions();
    }

    return scheduledLoop;
}

void FeasibleMemoryScheduler::scheduleComputeOps() {
    // preserve order of compute ops
    if (!getReadyLoopOps().empty()) {
        // schedule loops before other compute ops
        return;
    }

    std::map<FeasibleMemoryScheduler::QueueType, operationIdxType> loopNextIdx;
    for (size_t idx = 0; idx < _loopRegions.size(); idx++) {
        const auto& computeRegion = _loopRegions[idx];
        if (computeRegion.getLoopType() == LoopType::None) {
            continue;
        }
        if (_scheduledLoopRegionInd.count(idx)) {
            continue;
        }

        _log.trace("Loop region {0}", idx);

        // Check deps for 1st loop
        for (const auto& opInfo : computeRegion.schedulingLoop->loopBodies[0]) {
            // some deps can be added for compute op ordering
            if (opInfo.allocationType == AllocationType::COMPUTE) {
                const auto queueType = getQueueType(opInfo.opIdx);
                loopNextIdx[queueType] = opInfo.opIdx;
                break;
            }
        }
        break;
    }

    SmallVector<std::pair<operationIdxType, size_t>> scheduledOps;
    mlir::DenseSet<mlir::Value> buffersToAllocate;
    SmallVector<operationIdxType> computeOpIdxToSchedule;

    // Ops with dependencies on other compute ops
    SmallVector<std::pair<operationIdxType, FeasibleMemoryScheduler::QueueType>> computeOps;
    // Ops which depend on constants or function inputs have lower priority
    SmallVector<std::pair<operationIdxType, FeasibleMemoryScheduler::QueueType>> noInputDepComputeOps;

    // Prepare status and snapshots for active spilling and fragmentation optimization

    ScheduleStateSnapshot snap(*this, _activelySpillForPrefetching);
    resetScheduleComputeOpsDeltas();

    // find compute ops to schedule
    for (auto& queue : _computeOpOrder) {
        auto firstOpInQueue = queue.second.begin();
        if (firstOpInQueue == queue.second.end()) {
            // no ops on queue left
            continue;
        }
        _log.trace("Next compute op: '{0}'", *firstOpInQueue);
        if (_readyComputeOps.find(*firstOpInQueue) == _readyComputeOps.end()) {
            // operation not ready
            _log.nest().trace("Not ready {0}", *firstOpInQueue);
            continue;
        }
        if (!unscheduledOpsOnQueue(queue.first)) {
            // need to unschedule ops on queue before scheduling
            _log.nest().trace("Not unscheduled {0}", *firstOpInQueue);
            continue;
        }
        if (loopNextIdx.count(queue.first) && loopNextIdx[queue.first] < *firstOpInQueue) {
            // loop will be scheduled before
            _log.nest().trace("Loop before {0}", *firstOpInQueue);
            continue;
        }

        if (isNoInputDepComputeOp(*firstOpInQueue)) {
            _log.trace("Compute op '{0}' depends only on constants so added to lower prio queue", *firstOpInQueue);
            noInputDepComputeOps.push_back({*firstOpInQueue, queue.first});
        } else {
            // Store compute ops with dependencies to higher prio queue
            computeOps.push_back({*firstOpInQueue, queue.first});
        }
    }

    SmallVector<std::pair<operationIdxType, FeasibleMemoryScheduler::QueueType>> computeOpsToErase;

    // Check if compute ops satisfy buffer allocation constraints and select them for scheduling
    auto selectOpsToSchedule = [&](SmallVector<std::pair<operationIdxType, FeasibleMemoryScheduler::QueueType>>& ops) {
        for (auto opIter = ops.begin(); opIter != ops.end();) {
            auto operationBuffers = getBuffersToAllocateForOp(opIter->first);
            operationBuffers.insert(buffersToAllocate.begin(), buffersToAllocate.end());
            _log.trace("Try to schedule compute op: '{0}'", opIter->first);
            if (!canAllocBuffers(operationBuffers)) {
                // operation does not fit in memory
                _log.nest().trace("Operation does not fit in memory: '{0}'", opIter->first);
                opIter++;
                continue;
            }

            // op will be scheduled
            buffersToAllocate = std::move(operationBuffers);
            computeOpIdxToSchedule.push_back(opIter->first);
            _log.trace("Compute op to schedule: '{0}'", opIter->first);
            computeOpsToErase.push_back(*opIter);
            opIter = ops.erase(opIter);
        }
    };

    selectOpsToSchedule(computeOps);
    if (computeOps.empty()) {
        // Schedule noInputDepComputeOps as close as possible to its consumers.
        // Do not schedule them if there are other ready ops which do not fit into CMX yet.
        selectOpsToSchedule(noInputDepComputeOps);
    }

    // schedule compute ops
    for (auto& computeOpIdx : computeOpIdxToSchedule) {
        _log.trace("Scheduling compute op: '{0}'", computeOpIdx);

        auto opCycleEnd = scheduleOp(computeOpIdx);
        _readyComputeOps.erase(computeOpIdx);
        scheduledOps.push_back(std::make_pair(computeOpIdx, opCycleEnd));
    }

    // Return true if an alive buffer can be found to spill to support prefetching
    // and set _evictionCandidateToSupportPrefetch
    auto foundProperSpillCandidateToSupportPrefetch = [&]() {
        auto aliveBuffers = snap.scan.handler().getAliveValues();
        if (aliveBuffers.empty()) {
            return false;
        }

        // `originalBufferProducer` is the stage before running the current `scheduleComputeOps`.
        // So the changes in `_bufferProducer` is reverted.
        // And eviction candidate need to be selected in the alive buffers before the current scheduling
        // because the current scheduling will be redone with the eviction
        auto originalBufferProducer = _bufferProducer;
        for (auto& val : _originalBufferProducersFromScheduleComputeOps) {
            originalBufferProducer[val.first] = val.second;
        }

        for (auto& val : _newBufferProducersFromScheduleComputeOps) {
            originalBufferProducer.erase(val);
        }

        _evictionCandidateToSupportPrefetch =
                chooseCandidateForEvictionToSupportPrefetch(aliveBuffers, originalBufferProducer, snap.scan);
        if (!_evictionCandidateToSupportPrefetch.has_value()) {
            return false;
        }

        return true;
    };

    auto restoreStateBeforeScheduling = [&]() {
        // restore state before scheduling compute ops
        snap.rollback();

        for (auto& val : _originalBufferProducersFromScheduleComputeOps) {
            _bufferProducer[val.first] = val.second;
        }

        for (auto& val : _newBufferProducersFromScheduleComputeOps) {
            _bufferProducer.erase(val);
        }

        for (auto& val : _originalOpIdxEndCycleMapFromScheduleComputeOps) {
            _opIdxEndCycleMap[val.first] = val.second;
        }

        for (auto& val : _newOpsFromScheduleComputeOps) {
            _opIdxEndCycleMap.erase(val);
        }

        for (auto& val : _originalBufferLastCycleUseFromScheduleComputeOps) {
            _bufferLastCycleUse[val.first] = val.second;
        }

        for (auto& val : _newBufferLastCycleUsesFromScheduleComputeOps) {
            _bufferLastCycleUse.erase(val);
        }
    };

    // prefetch data ops
    if (!computeOpIdxToSchedule.empty()) {
        // If fragmentation is detected, _prefetchFailedDueToFragmentation is changed to true
        // And the active spilling logic will be triggered to find an eviction candidate to support prefetching
        prefetchOps(scheduledOps, buffersToAllocate, _activelySpillForPrefetching);
        if (_prefetchFailedDueToFragmentation) {
            // If prefetchOps failed due to fragmentation, try to find a proper spill candidate
            // If candidate found (_prefetchFailedDueToFragmentation is still true),
            // the candidate is saved in `_evictionCandidateToSupportPrefetch`
            _prefetchFailedDueToFragmentation = foundProperSpillCandidateToSupportPrefetch();
        }

        if (_prefetchFailedDueToFragmentation) {
            // If prefetchOps failed due to fragmentation, but successfully found a spill candidate,
            // restore state before scheduling compute ops and early return
            // so the spilling can be performed by `forceSpillingForPrefetch` on `_evictionCandidateToSupportPrefetch`
            // and prefetching should work after that

            ++_prefetchFragmentationFailureFixedCount;
            restoreStateBeforeScheduling();
            return;
        }

        for (auto& val : buffersToAllocate) {
            _scan.handler().markAsAlive(val);
            _log.nest().trace("5. Mark as alive '{0}'", val);
        }
    }

    for (auto opIter = computeOpsToErase.begin(); opIter != computeOpsToErase.end(); opIter++) {
        auto& scheduledQueue = _computeOpOrder[opIter->second];
        scheduledQueue.erase(scheduledQueue.begin());
    }

    // find DMA ops to schedule
    SmallVector<operationIdxType> DMAOpIdxToSchedule;
    for (auto& readyOpIdx : _readyDMAOps) {
        auto operationBuffers = getBuffersToAllocateForOp(readyOpIdx);
        operationBuffers.insert(buffersToAllocate.begin(), buffersToAllocate.end());
        if (!canAllocBuffers(operationBuffers)) {
            continue;
        }

        buffersToAllocate = std::move(operationBuffers);
        DMAOpIdxToSchedule.push_back(readyOpIdx);
    }

    // schedule DMA ops
    SmallVector<operationIdxType> readyOps = {};
    for (auto& DMAOpIdx : DMAOpIdxToSchedule) {
        _log.trace("Scheduling DMA op: '{0}'", DMAOpIdx);

        auto opCycleEnd = scheduleOp(DMAOpIdx);
        _readyDMAOps.erase(DMAOpIdx);
        scheduledOps.push_back(std::make_pair(DMAOpIdx, opCycleEnd));

        auto newReadyOps = reduceInDegreeOfAdjacentOperations(DMAOpIdx);
        readyOps.insert(readyOps.end(), newReadyOps.begin(), newReadyOps.end());
    }
    // unlock DMA copy back in ops to be prefetched, so we can achieve:
    // | DPU | DMA-out | DMA-in |
    // | -------- SW ---------- |
    distributeReadyOps(readyOps);

    // prefetch data ops - activation reads
    if (!DMAOpIdxToSchedule.empty()) {
        prefetchOps(scheduledOps, buffersToAllocate, false);
    }

    // TODO: E93150 gather all ops to schedule and sort by which-ever can be scheduled earlier

    // allocate buffers
    sortAndAllocateBuffers(buffersToAllocate);
}

void FeasibleMemoryScheduler::scheduleNonComputeOps() {
    mlir::DenseSet<mlir::Value> buffersToAllocate;

    // schedule operation not belonging to main network compute chain as soon as they become
    // ready so that they execute in the next available cycle since they are not prefetched
    for (auto& readyOpIdx : llvm::make_early_inc_range(_nonComputeChainOps)) {
        // Scheduling such operations can only happen once all input dependencies
        // (both data and compute ops) have already been executed. This is different
        // to standard compute op which as part of its scheduling can force scheduling
        // of needed data ops
        bool areDepsReady = true;
        for (const auto& dep : _depsInfo.getOpDeps(readyOpIdx)) {
            if (_spillBufferMap.find(dep) != _spillBufferMap.end()) {
                areDepsReady = false;
                break;
            }
        }
        if (!areDepsReady) {
            continue;
        }

        auto operationBuffers = getBuffersToAllocateForOp(readyOpIdx);
        operationBuffers.insert(buffersToAllocate.begin(), buffersToAllocate.end());
        if (!operationBuffers.empty()) {
            VPUX_THROW("Non-empty profiling buffers for {0}", readyOpIdx);
        }

        if (!canAllocBuffers(operationBuffers)) {
            continue;
        }

        buffersToAllocate = std::move(operationBuffers);

        _log.trace("Scheduling non compute chain op: '{0}'", readyOpIdx);
        scheduleOp(readyOpIdx);
        _nonComputeChainOps.erase(readyOpIdx);
    }

    // allocate buffers
    sortAndAllocateBuffers(buffersToAllocate);
}

void FeasibleMemoryScheduler::insertInOpIdxCycleEndMap(const operationIdxType& opIdx, const size_t& endCycle) {
    auto mapItr = _opIdxEndCycleMap.find(opIdx);
    if (mapItr == _opIdxEndCycleMap.end()) {
        _opIdxEndCycleMap[opIdx] = endCycle;
        _newOpsFromScheduleComputeOps.insert(opIdx);
        return;
    }

    if (mapItr->second < endCycle) {
        if (_originalOpIdxEndCycleMapFromScheduleComputeOps.find(opIdx) ==
            _originalOpIdxEndCycleMapFromScheduleComputeOps.end()) {
            _originalOpIdxEndCycleMapFromScheduleComputeOps[opIdx] = mapItr->second;
        }
        _opIdxEndCycleMap[opIdx] = endCycle;
    }
}

size_t FeasibleMemoryScheduler::getEarliestComputeBeginCycle(operationIdxType opIdx) {
    auto queueAndCycle = getCurrentCycleAndExecutorInstanceMask(opIdx);
    auto earliestComputeBeginCycle = queueAndCycle.cycle;
    // precondition: all producers of used buffers scheduled
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    const auto usedBufs = _liveRangeInfo.getUsedBuffers(op);
    for (auto& buffer : usedBufs) {
        if (_bufferProducer.find(buffer) != _bufferProducer.end()) {
            // use cycle end of latest writing op
            earliestComputeBeginCycle = std::max(_opIdxEndCycleMap[_bufferProducer[buffer]], earliestComputeBeginCycle);
        }
    }
    return earliestComputeBeginCycle;
}

void FeasibleMemoryScheduler::evictActiveOp(EvictionCandidate evictionCandidate) {
    VPUX_THROW_UNLESS(_opIdxEndCycleMap.find(evictionCandidate.bufferWriterIdx_) != _opIdxEndCycleMap.end(),
                      "Attempt to evict a non-scheduled operation");

    _readySpilledOps[evictionCandidate.buffer_] = evictionCandidate.bufferWriterIdx_;
    _spillBufferMap[evictionCandidate.bufferWriterIdx_].insert(evictionCandidate.buffer_);

    _log.nest().trace("Mark dynamically spilled buffer as dead, '{0}'", evictionCandidate.buffer_);
    _scan.handler().markAsDead(evictionCandidate.buffer_);
    _scan.handler().markAsDynamicSpill(evictionCandidate.buffer_);

    _log.nest().trace("Free non alive buffers");
    _scan.freeDeadRanges();
}

size_t FeasibleMemoryScheduler::evictionPriority(operationIdxType writerOpIdx, mlir::Value buffer) {
    // TODO: E#21936 add other conditions such as:
    // pipelined, multiple out-degree (prefetch)

    // Eviction priority (highest evicted first):
    // (0) - buffers which are CMX contactable
    // (1) - timestamp op buffers
    // (2) - buffers which are result of computeOp
    // (3) - buffers which are result of dataOp

    for (auto bufferUser : buffer.getUsers()) {
        if (mlir::isa<VPUIP::ConcatViewOp>(bufferUser)) {
            // buffer CMX contactable
            return 0;
        }
    }

    if (!_isDataOp[writerOpIdx]) {
        return 2;
    }

    return 3;
}

size_t FeasibleMemoryScheduler::getOpBufferOutputIdx(operationIdxType opIdx, mlir::Value buffer) {
    size_t outputIdx = 0;

    // Get asyncExecOp result corresponding to given buffer
    auto asyncExecOp = _depsInfo.getExecuteOpAtIndex(opIdx);

    for (auto& outBuffer : _liveRangeInfo.getOutputBuffers(asyncExecOp)) {
        if (outBuffer == buffer) {
            return outputIdx;
        }
        outputIdx++;
    }

    VPUX_THROW("Unable to find async.execOp (opIdx - '{0}') result corresponding to buffer '{1}'", opIdx, buffer);
}

operationIdxType FeasibleMemoryScheduler::getEarliestConsumerIdx(operationIdxType opIdx) {
    auto earliestConsumerIdx = std::numeric_limits<operationIdxType>::max();
    for (const auto& consumerIdx : _depsInfo.getConsumerOps(opIdx)) {
        if (!VPUIP::VPUIPDialect::isComputeExecutorKind(getExecutorType(consumerIdx))) {
            continue;
        }
        if (_opIdxEndCycleMap.find(consumerIdx) != _opIdxEndCycleMap.end()) {
            continue;
        }

        earliestConsumerIdx = std::min(earliestConsumerIdx, consumerIdx);
    }
    return earliestConsumerIdx;
}

// Choose the best candidate for eviction to support prefetching
// The candidate should be the smallest buffer that is not currently in use
// and whose eviction would free enough memory to allow prefetching
// For example, next buffer size to prefetch is 100, but there are only 80 continuous free memory due to fragmentation
//      Total cmx size is 200. The alive buffers are:
//          A [0, 100], B [180, 190], C [210, 215], D [220, 300]
//      Gaps are:
//          80: [100, 180], 20: [190, 210], 5: [215, 220]
//      Buffer size of 100 can't be prefetched due to fragmentation, so C, B, D, A are evicted in order to see if
//      the eviction could fix the fragmentation Eviction of B succeed so B is chosen to be eviction candidate for
//      prefetch
std::optional<FeasibleMemoryScheduler::EvictionCandidate>
FeasibleMemoryScheduler::chooseCandidateForEvictionToSupportPrefetch(
        mlir::DenseSet<mlir::Value>& aliveBuffers, mlir::DenseMap<mlir::Value, operationIdxType>& bufferProducer,
        LinearScan<mlir::Value, LinearScanHandler>& scan) {
    std::optional<FeasibleMemoryScheduler::EvictionCandidate> evictionCandidate = std::nullopt;

    // Sort active buffers by priority (lower priority evicted first)
    auto sortedAliveBuffers = sortUsedBuffers(aliveBuffers);

    // Record the smallest eviction candidate
    size_t evictionCandidateSize = 0;

    // Iterate over all alive buffers to find the best candidate for eviction
    for (const auto buffer : llvm::reverse(sortedAliveBuffers)) {
        // Ensure the buffer has been scheduled
        VPUX_THROW_UNLESS(bufferProducer.find(buffer) != bufferProducer.end(),
                          "Buffer not scheduled yet, invalid eviction candidate");

        auto executeOpIdx = bufferProducer[buffer];
        // Check if the buffer is in use
        bool bufferInUse = false;
        for (auto& nextOp : llvm::make_early_inc_range(_cycleEndHeap)) {
            if (nextOp.op_ == executeOpIdx) {
                bufferInUse = true;
                break;
            }
        }

        if (bufferInUse) {
            // Skip buffers that are currently in use
            continue;
        }

        auto size = scan.handler().getSize(buffer);
        if (evictionCandidate.has_value() && (evictionCandidateSize <= size)) {
            // Skip larger or equal sized buffers if a smaller candidate is already found
            continue;
        }

        // Create a temporary scan and mark the current buffer as dead
        // In tempScan the aliveBuffer and liveRange is temporarily changed to simulate the prefetching with eviction
        // The original scan shouldn't be changed so it can be roll back to the stage
        auto tempScan = scan;
        // Simulate the eviction by marking the buffer as dead in tempScan
        tempScan.handler().markAsDead(buffer);
        tempScan.freeDeadRanges();

        // Create a temporary set of operation buffers including the current buffer
        auto tempOperationBuffers = _fragmentedBuffers;
        tempOperationBuffers.insert(buffer);

        auto canAllocBuffers = [&]() {
            if (_optimizeFragmentation) {
                // sort to minimize fragmentation
                auto sortedBuffers = sortUsedBuffers(tempOperationBuffers);
                // are resources available and can be allocated
                return tempScan.canAlloc(sortedBuffers);
            }
            return tempScan.canAlloc(tempOperationBuffers);
        };
        // Check if resources are available and can be allocated with eviction
        // If yes, record the eviction candidate with smaller size
        if (canAllocBuffers()) {
            auto priority = evictionPriority(executeOpIdx, buffer);
            auto earliestConsumerIdx = getEarliestConsumerIdx(executeOpIdx);
            // in special case of multiple output buffers store output idx
            auto outputIdx = getOpBufferOutputIdx(executeOpIdx, buffer);
            evictionCandidate = EvictionCandidate(priority, earliestConsumerIdx, size, executeOpIdx, outputIdx, buffer);
            evictionCandidateSize = size;
        }
    }

    return evictionCandidate;
}

FeasibleMemoryScheduler::EvictionCandidate FeasibleMemoryScheduler::chooseCandidateForEviction(
        const mlir::DenseSet<mlir::Value>& aliveBuffers) {
    // Check if last scheduled op was a SPILL-WRITE. If yes then this is a direct subsequent spill
    // and eviction candidates can be picked up from cache which was prepared during previous search
    // for spill write buffer
    if (!_evictionCandidatesCache.empty() && _scheduledOps.back().isSpillWrite()) {
        auto evictionCandidate = *_evictionCandidatesCache.begin();
        _evictionCandidatesCache.erase(_evictionCandidatesCache.begin());
        return evictionCandidate;
    }
    _evictionCandidatesCache.clear();
    // sort buffers using eviction priority
    for (const auto& buffer : aliveBuffers) {
        VPUX_THROW_UNLESS(_bufferProducer.find(buffer) != _bufferProducer.end(),
                          "Buffer not scheduled yet, invalid eviction candidate");
        auto executeOpIdx = _bufferProducer[buffer];
        auto priority = evictionPriority(executeOpIdx, buffer);
        auto earliestConsumerIdx = getEarliestConsumerIdx(executeOpIdx);
        auto size = _scan.handler().getSize(buffer);
        // in special case of multiple output buffers store output idx
        auto outputIdx = getOpBufferOutputIdx(executeOpIdx, buffer);
        _evictionCandidatesCache.insert(
                EvictionCandidate(priority, earliestConsumerIdx, size, executeOpIdx, outputIdx, buffer));
    }

    // Get eviction candidate with highest priority (beginning of set)
    // Rest will be left in a cache in case of subsequent spilling
    auto evictionCandidate = *_evictionCandidatesCache.begin();
    _evictionCandidatesCache.erase(_evictionCandidatesCache.begin());

    return evictionCandidate;
}

size_t FeasibleMemoryScheduler::performEvictionAndScheduling(const EvictionCandidate& evictionCandidate,
                                                             bool alignExecutorsFlag) {
    auto spillType = EOpType::IMPLICIT_SPILL_WRITE_OP;

    // Understand if spilling happened due to fragmentation
    // This is the case where operation CMX demand is smaller than
    // total free CMX size but because free contiguous slots are not
    // large enough operation buffers cannot be allocated
    if (_enableScheduleStatistics) {
        size_t freeCmx = _scan.totalFreeSize();
        bool spillingDueToFragmentation = false;

        for (auto& readyOp : _readyComputeOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            if (opTotalSize <= freeCmx) {
                if (!spillingDueToFragmentation) {
                    spillingDueToFragmentation = true;
                    break;
                }
            }
        }
        for (auto& readyOp : _readyDMAOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            if (opTotalSize <= freeCmx) {
                if (!spillingDueToFragmentation) {
                    spillingDueToFragmentation = true;
                    break;
                }
            }
        }

        if (spillingDueToFragmentation) {
            spillType = EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP;
        }

        // TODO: In future scheduler could try to reorder buffers to make necessary
        // space for next operation in case spilling happened due to CMX fragmentation
    }

    _log.nest().trace("Candidate selected for eviction '{0}' '{1}'", evictionCandidate.bufferWriterIdx_,
                      evictionCandidate.buffer_);

    // free the memory space by freeing the op output buffer
    evictActiveOp(evictionCandidate);
    _log.nest().trace("Candidate evicted and spilled");

    // consider spilling operation cycle end
    // TODO: consider last used cycle and next available cycle for spill to avoid stall
    const auto depEndCycle = _bufferLastCycleUse[evictionCandidate.buffer_];
    const auto queueAndCycle =
            getCurrentCycleAndExecutorInstanceMaskForSpill(evictionCandidate.buffer_, spillType, depEndCycle);
    // find operation end cycle
    const auto opCycleCost = spilledOperationCycleCost(evictionCandidate.buffer_);
    const auto nextAvailableCycle = queueAndCycle.cycle + opCycleCost;

    // add with a spilled write state
    pushToCycleBeginHeap(HeapElement(evictionCandidate.bufferWriterIdx_, queueAndCycle, opCycleCost, spillType,
                                     evictionCandidate.buffer_));
    // update current cycle directly
    updateCurrentCycleForExecutor(queueAndCycle.queueType, queueAndCycle.execMask, nextAvailableCycle);

    // memory resource freed, need to align executors to only allocate in future cycles
    if (alignExecutorsFlag) {
        alignExecutors(nextAvailableCycle);
    }

    return nextAvailableCycle;
}

void FeasibleMemoryScheduler::forceSpillingForPrefetch() {
    _log.trace("Unable to prefetch due to fragmentation, forcing dynamic spill");
    // select a candidate op to be spilled
    VPUX_THROW_UNLESS(_evictionCandidateToSupportPrefetch.has_value(), "Failed to find eviction candidate");
    auto evictionCandidateValue = _evictionCandidateToSupportPrefetch.value();
    performEvictionAndScheduling(evictionCandidateValue);
}

void FeasibleMemoryScheduler::forceScheduleActiveOpEviction() {
    _log.trace("Unable to schedule an operation, forcing dynamic spill");

    // retrieve the alive buffers
    auto aliveBuffers = _scan.handler().getAliveValues();
    size_t freeCmx = _scan.totalFreeSize();
    if (aliveBuffers.empty()) {
        _log.error("Scheduler cannot schedule anything and there is no buffer to spill");
        _log.error("Next operations to schedule:");
        for (auto& nextOp : _computeOpOrder) {
            _log.nest().error("opIdx: {0}, on: {1}", *nextOp.second.begin(), nextOp.first.execKind);
        }
        _log.error("Ready operations:");
        for (auto& readyOp : _readyComputeOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            auto execOp = _depsInfo.getExecuteOpAtIndex(readyOp);
            _log.nest().error(
                    "readyComputeOp: opIdx: {0}, size demand: {1}, available free CMX: {2}, name: {3}, op: {4}, ",
                    readyOp, opTotalSize, freeCmx, execOp.getLoc(), execOp);
        }
        for (auto& readyOp : _readyDMAOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            auto execOp = _depsInfo.getExecuteOpAtIndex(readyOp);
            _log.nest().error("readyDMAOp: opIdx: {0}, size demand: {1}, available free CMX: {2}, name: {3}, op: {4}, ",
                              readyOp, opTotalSize, freeCmx, execOp.getLoc(), execOp);
        }
        for (auto& readyOp : _readyDataOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            auto execOp = _depsInfo.getExecuteOpAtIndex(readyOp);
            _log.nest().error(
                    "readyDataOp: opIdx: {0}, size demand: {1}, available free CMX: {2}, name: {3}, op: {4}, ", readyOp,
                    opTotalSize, freeCmx, execOp.getLoc(), execOp);
        }
        for (auto& readyOp : _nonComputeChainOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            auto execOp = _depsInfo.getExecuteOpAtIndex(readyOp);
            _log.nest().error(
                    "nonComputeChainOp: opIdx: {0}, size demand: {1}, available free CMX: {2}, name: {3}, op: {4}, ",
                    readyOp, opTotalSize, freeCmx, execOp.getLoc(), execOp);
        }

        // select a candidate op to be spilled
        cleanUpAndLogSchedule(_scheduledOps);
        VPUX_THROW("Scheduler failure, cannot schedule anything and there is no buffer to spill");
    }

    auto evictionCandidate = chooseCandidateForEviction(aliveBuffers);
    performEvictionAndScheduling(evictionCandidate);
}

void FeasibleMemoryScheduler::createBufferAsyncIdxMap() {
    auto populateMap = [&](mlir::Value buffer, size_t operationIdx,
                           mlir::DenseMap<mlir::Value, SmallVector<size_t>>& bufferOpIdxMap) -> bool {
        auto insertedPair = bufferOpIdxMap.insert({buffer, {operationIdx}});
        if (!insertedPair.second) {
            bufferOpIdxMap[buffer].push_back(operationIdx);
        }
        return true;
    };
    for (auto& asyncDepsPair : _outDegreeTable) {
        auto executeOp = _depsInfo.getExecuteOpAtIndex(asyncDepsPair.first);

        for (auto& buffer : _liveRangeInfo.getOutputBuffers(executeOp)) {
            if (!populateMap(buffer, asyncDepsPair.first, _bufferOpIdxMap)) {
                continue;
            }
        }
    }
}

void FeasibleMemoryScheduler::populateScheduledOps(const HeapElement& scheduledOp) {
    SmallVector<IntervalInfo> outputIntervals;
    SmallVector<IntervalInfo> inputIntervals;
    // store scheduled information
    if (scheduledOp.isSpillWriteOp()) {
        // special case for a spill write with deallocation
        IntervalInfo interval;
        // retrieve and store operation addresses
        interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(scheduledOp.spillBuffer_));
        interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(scheduledOp.spillBuffer_));
        interval.buffer_ = scheduledOp.spillBuffer_;
        // SPILL WRITE has only input resource
        inputIntervals.push_back(interval);
        // deallocate only after addresses stored
        _log.nest().trace("Deallocate, '{0}'", scheduledOp.spillBuffer_);
        _scan.handler().deallocate(scheduledOp.spillBuffer_);
    } else if (scheduledOp.isSpillReadOp()) {
        IntervalInfo interval;
        // retrieve and store operation addresses
        interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(scheduledOp.spillBuffer_));
        interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(scheduledOp.spillBuffer_));
        interval.buffer_ = scheduledOp.spillBuffer_;
        outputIntervals.push_back(interval);
    } else {
        auto execOp = _depsInfo.getExecuteOpAtIndex(scheduledOp.op_);
        auto addIntervals = [&](bool isInput, ValueOrderedSet buffers, SmallVector<IntervalInfo>& intervals) {
            for (auto& buffer : buffers) {
                if ((isInput && (_bufferProducer.find(buffer) == _bufferProducer.end())) ||
                    (!isInput && (!scheduledOp.isOriginalOp() && buffer != scheduledOp.spillBuffer_))) {
                    continue;
                }
                IntervalInfo interval;
                // retrieve and store operation addresses
                interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(buffer));
                interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(buffer));
                interval.buffer_ = buffer;
                intervals.push_back(interval);
            }
        };

        addIntervals(true, _liveRangeInfo.getInputBuffers(execOp), inputIntervals);
        addIntervals(false, _liveRangeInfo.getOutputBuffers(execOp), outputIntervals);
    }
    // populate the struct fields
    ScheduledOpInfo scheduled;
    scheduled.op_ = scheduledOp.op_;
    scheduled.opType_ = scheduledOp.opType_;
    scheduled.outputResourceInfo_ = std::move(outputIntervals);
    scheduled.inputResourceInfo_ = std::move(inputIntervals);
    scheduled.cycleBegin_ = scheduledOp.cycleBegin_;
    scheduled.cycleEnd_ = scheduledOp.cycleEnd_;
    scheduled.isDataOp_ = _isDataOp[scheduledOp.op_];
    scheduled.freeCmx_ = _scan.totalFreeSize();
    if (scheduledOp.isSpillOp()) {
        scheduled.queueType.execKind = config::ExecutorKind::DMA_NN;
        if (scheduledOp.isSpillWriteOp()) {
            scheduled.queueType.id = getDMAQueueIdEncoding(_memKind, _archKind);
        } else {
            scheduled.queueType.id = getDMAQueueIdEncoding(_secondLvlMemKind, _archKind);
        }
    } else {
        scheduled.queueType = getQueueType(scheduledOp.op_);
    }
    // scheduled.queueType = scheduledOp.isSpillOp() ? config::ExecutorKind::DMA_NN : getExecutorType(scheduledOp.op_);
    scheduled.executorInstanceMask = scheduledOp.executorInstanceMask_;
    _scheduledOps.push_back(scheduled);
    insertInOpIdxCycleEndMap(scheduled.op_, scheduled.cycleEnd_);
    _log.trace("Scheduled op: '{0}' during cycles: {1} -> {2}, of type: {3}", scheduled.op_, scheduled.cycleBegin_,
               scheduled.cycleEnd_, scheduled.opTypeName());
}

void FeasibleMemoryScheduler::clearLists() {
    _readyComputeOps.clear();
    _readyDMAOps.clear();
    _readyDataOps.clear();
}

bool FeasibleMemoryScheduler::init() {
    _log.trace("Feasible Memory Scheduler init()");
    _depsInfo.buildConsMap();

    // compute op in/out degree
    _inDegreeTable = _depsInfo.calculateOpInDegreeTable();
    _outDegreeTable = _depsInfo.calculateOpOutDegreeTable();

    // retrieve output ops (ops with no out-degree)
    for (auto& entry : _outDegreeTable) {
        if (entry.second == 0) {
            _outputOps.insert(entry.first);
        }
    }

    identifyDataOps();

    size_t level = 0;
    _opLevelVec.resize(_inDegreeTable.size(), 0);
    for (size_t computeOpIdx = 0; computeOpIdx < _inDegreeTable.size(); ++computeOpIdx) {
        if (_isDataOp[computeOpIdx]) {
            continue;
        }

        const auto queueType = getQueueType(computeOpIdx);
        if (!VPUIP::VPUIPDialect::isComputeExecutorKind(queueType.execKind)) {
            continue;
        }

        _opLevelVec[computeOpIdx] = level;
        for (const auto& depInd : _depsInfo.getOpDeps(computeOpIdx)) {
            if (_opLevelVec[depInd] != 0) {
                continue;
            }
            _opLevelVec[depInd] = level;
        }

        if (_loopRegionInd.count(computeOpIdx)) {
            ++level;
            continue;
        }

        _log.trace("Compute op order: '{0}'", computeOpIdx);

        _computeOpOrder[queueType].push_back(computeOpIdx);
        ++level;
    }

    clearLists();
    // TODO: check if input is dag
    initializeReadyLists();
    createBufferAsyncIdxMap();
    schedulingLoop();

    return true;
}

void FeasibleMemoryScheduler::schedulingLoop() {
    // scheduling loop, loop until all output ops are scheduled
    while (!_outputOps.empty()) {
        // Loop regions have higher scheduling priority
        // Always try to schedule loop first
        auto loopScheduled = scheduleLoopRegions();

        if (!_cycleBeginHeap.empty()) {
            _log.nest().trace("0. MOVE FROM CYCLE BEGIN TO CYCLE END HEAP");
            // move ops from cycle begin to cycle end heap
            // - populate ScheduledOpInfoVec with op info
            // - decrease any scheduled output ops
            moveFromCycleBeginToCycleEndHeap();
        } else {
            _log.nest().trace("1. UNSCHEDULE OPS FROM CYCLE END HEAP");
            // 1. unschedule all operations from cycle end heap
            //  - free memory of consumed buffers
            //  - unlock new ready operations
            unscheduleAllCompletingOps();

            _log.nest().trace("2. SCHEDULE OPS TO CYCLE BEGIN HEAP");
            // 2. schedule ready operations
            //  - allocate compute operations along with data dependencies
            //  - update executor pipelines
            // 2.1. schedule operation not belonging to main network compute chain
            scheduleNonComputeOps();
            // 2.2 schedule compute operations
            scheduleComputeOps();
            // 2.3 schedule loop regions
            loopScheduled |= scheduleLoopRegions();

            if (_prefetchFailedDueToFragmentation) {
                // optional 2.3 only enabled when _activelySpillForPrefetching is true
                // if prefetch failed due to fragmentation and `_evictionCandidateToSupportPrefetch` was found
                // force spilling of the candidate and try to prefetch again
                _log.nest().trace("DYNAMIC SPILL REQUIRED FOR PREFETCH");
                forceSpillingForPrefetch();
            }

            // 3. if no operation was added to cycle begin heap after scheduling
            //  - unable to schedule an operation, perform dynamic spill
            if (!loopScheduled && _cycleBeginHeap.empty() && _cycleEndHeap.empty()) {
                _log.nest().trace("3. DYNAMIC SPILL REQUIRED: FORCE DYNAMIC SPILL");
                forceScheduleActiveOpEviction();
            }
        }
    }
}

void FeasibleMemoryScheduler::cleanUpAndLogSchedule(ScheduledOpInfoVec& scheduledOps) {
    // schedule quality based on cycles (cycles start from 1)
    std::map<QueueType, SmallVector<int64_t>> dpuOrDmaQueuesCycles;

    int64_t totalCycles = 0;

    _log.setName("feasible-schedule");
    _log = _log.nest();
    for (const auto& op : scheduledOps) {
        auto execOp = _depsInfo.getExecuteOpAtIndex(op.op_);
        std::string inputResourceInfo = "<none>";
        std::string outputResourceInfo = "<none>";
        std::string executorInstanceInfo = "";
        auto channelTypeAsString = op.queueType.execKind == config::ExecutorKind::DMA_NN
                                           ? VPUIP::getDMAChannelTypeAsString(op.queueType.id, _archKind)
                                           : "";

        if (op.queueType.execKind == config::ExecutorKind::DMA_NN && channelTypeAsString.size() > 0) {
            executorInstanceInfo += "_" + channelTypeAsString;
        }

        if (_executorPipelines[op.queueType].size() > 1) {
            executorInstanceInfo += " [";
            bool addComma = false;
            for (auto execInd : op.executorInstanceMask.set_bits()) {
                if (addComma) {
                    executorInstanceInfo += ",";
                }
                executorInstanceInfo += std::to_string(execInd);
                addComma = true;
            }

            executorInstanceInfo += "]";
        }

        if (op.hasActiveInputResource()) {
            inputResourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfInputResources(); resourceIdx++) {
                if (op.isActiveInputResource(resourceIdx)) {
                    inputResourceInfo +=
                            "[" + std::to_string(op.beginInputResource(resourceIdx)) + " " +
                            std::to_string(op.endInputResource(resourceIdx)) + "] size = " +
                            std::to_string((op.endInputResource(resourceIdx) - op.beginInputResource(resourceIdx))) +
                            ", ";
                }
            }
        }

        if (op.hasActiveOutputResource()) {
            outputResourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfOutputResources(); resourceIdx++) {
                if (op.isActiveOutputResource(resourceIdx)) {
                    outputResourceInfo +=
                            "[" + std::to_string(op.beginOutputResource(resourceIdx)) + " " +
                            std::to_string(op.endOutputResource(resourceIdx)) + "] size = " +
                            std::to_string((op.endOutputResource(resourceIdx) - op.beginOutputResource(resourceIdx))) +
                            ", ";
                }
            }
        }

        auto cycleInfo = "cycles = " + std::to_string(op.cycleBegin_) + " -> " + std::to_string(op.cycleEnd_);
        _log.trace("op = '{0}'\t executor = '{1}{2}'\t type = '{3}'\t '{4}'\t inputs = '{5}' outputs = '{6}' \t "
                   "free = "
                   "'{7}'\t name = '{8}'",
                   op.op_, op.queueType.execKind, executorInstanceInfo, op.opTypeName(), cycleInfo, inputResourceInfo,
                   outputResourceInfo, op.freeCmx_, execOp.getLoc());

        if (op.queueType.execKind == config::ExecutorKind::DMA_NN ||
            op.queueType.execKind == config::ExecutorKind::DPU) {
            if (dpuOrDmaQueuesCycles.find(op.queueType) == dpuOrDmaQueuesCycles.end()) {
                dpuOrDmaQueuesCycles[op.queueType].assign(_executorPipelines[op.queueType].size(), 1);
            }

            auto& execCycles = dpuOrDmaQueuesCycles[op.queueType];
            for (auto execInst : op.executorInstanceMask.set_bits()) {
                auto cycleDiff = op.cycleBegin_ - execCycles[execInst];
                if (cycleDiff > 0) {
                    std::string execInstString = "";

                    if (op.queueType.execKind == config::ExecutorKind::DMA_NN && channelTypeAsString.size() > 0) {
                        execInstString += "_" + channelTypeAsString;
                    }

                    if (_executorPipelines[op.queueType].size() > 1) {
                        execInstString += " [";
                        execInstString += std::to_string(execInst);
                        execInstString += "]";
                    }

                    _log.nest().trace("--- {0}{1} STALL ({2} cycles) ---", op.queueType.execKind, execInstString,
                                      cycleDiff);
                }
                execCycles[execInst] = op.cycleEnd_;
            }
        }

        totalCycles = std::max(totalCycles, op.cycleEnd_);
    }
    _log = _log.unnest();
    _log.trace("Total Cycles = {0}", totalCycles);
    _log.setName("feasible-memory-scheduler-allocator");
}

FeasibleMemoryScheduler::ScheduledOpInfoVec FeasibleMemoryScheduler::generateSchedule() {
    // start with all buffers requiring allocation
    _scan.handler().markAllBuffersAsDead();

    // start the memory scheduler
    init();

    // sort the operations to be reflected by IR order
    llvm::sort(_scheduledOps.begin(), _scheduledOps.end(), [](const ScheduledOpInfo& op1, const ScheduledOpInfo& op2) {
        // first cycle begin
        if (op1.cycleBegin_ != op2.cycleBegin_) {
            return op1.cycleBegin_ < op2.cycleBegin_;
        }

        // second smaller tasks first
        if (op1.cycleEnd_ != op2.cycleEnd_) {
            return op1.cycleEnd_ < op2.cycleEnd_;
        }

        // operation index
        if (op1.op_ != op2.op_) {
            return op1.op_ < op2.op_;
        }

        // allow self comparison
        return false;
    });

    return _scheduledOps;
}

bool FeasibleMemoryScheduler::canAllocBuffersWithReservedRanges(
        SmallVector<std::pair<vpux::AddressType, vpux::AddressType>>& reservedRanges,
        mlir::DenseSet<mlir::Value>& buffersToAllocate) {
    if (_optimizeFragmentation) {
        // sort to minimize fragmentation
        auto sortedBuffers = sortUsedBuffers(buffersToAllocate);
        // are resources available and can be allocated
        return _scan.canAllocWithExcludedRegion(reservedRanges, sortedBuffers);
    }
    return _scan.canAllocWithExcludedRegion(reservedRanges, buffersToAllocate);
}

void FeasibleMemoryScheduler::sortAndAllocateBuffersWithReservedRanges(
        SmallVector<std::pair<vpux::AddressType, vpux::AddressType>>& reservedRanges,
        mlir::DenseSet<mlir::Value>& buffersToAllocate) {
    _log.nest().trace("Allocate memory for the alive buffers");
    for (auto& val : buffersToAllocate) {
        _log.nest().trace("3. Mark as alive '{0}'", val);
        _scan.handler().markAsAlive(val);
    }

    if (_optimizeFragmentation) {
        auto usedBuffers = sortUsedBuffers(buffersToAllocate);
        VPUX_THROW_UNLESS(_scan.allocWithExcludedRegion(reservedRanges, usedBuffers, false, Partitioner::Direction::Up),
                          "Failed to statically allocate '{0}' memory", _memKind);
    } else {
        VPUX_THROW_UNLESS(
                _scan.allocWithExcludedRegion(reservedRanges, buffersToAllocate, false, Partitioner::Direction::Up),
                "Failed to statically allocate '{0}' memory", _memKind);
    }
}

bool FeasibleMemoryScheduler::canAllocBuffers(mlir::DenseSet<mlir::Value>& buffersToAllocate) {
    if (_optimizeFragmentation) {
        // sort to minimize fragmentation
        auto sortedBuffers = sortUsedBuffers(buffersToAllocate);
        // are resources available and can be allocated
        return _scan.canAlloc(sortedBuffers);
    }
    return _scan.canAlloc(buffersToAllocate);
}

void FeasibleMemoryScheduler::sortAndAllocateBuffers(mlir::DenseSet<mlir::Value>& buffersToAllocate) {
    if (buffersToAllocate.empty()) {
        return;
    }
    _log.nest().trace("Allocate memory for the alive buffers");
    for (auto& val : buffersToAllocate) {
        _log.nest().trace("3. Mark as alive '{0}'", val);
        _scan.handler().markAsAlive(val);
    }

    if (_optimizeFragmentation) {
        auto usedBuffers = sortUsedBuffers(buffersToAllocate);
        VPUX_THROW_UNLESS(_scan.alloc(usedBuffers, false, Partitioner::Direction::Up),
                          "Failed to statically allocate '{0}' memory", _memKind);
    } else {
        VPUX_THROW_UNLESS(_scan.alloc(buffersToAllocate, false, Partitioner::Direction::Up),
                          "Failed to statically allocate '{0}' memory", _memKind);
    }
}

void FeasibleMemoryScheduler::printFragmentFixCountLog() const {
    _log.info("[FeasibleAllocation statistics] fragmentation prefetch count {0}, fixed count {1}",
              _prefetchFragmentationFailureCount, _prefetchFragmentationFailureFixedCount);
}

bool FeasibleMemoryScheduler::canAllocBuffersWithReservedSize(mlir::DenseSet<mlir::Value>& buffersToAllocate,
                                                              vpux::AddressType reserve,
                                                              vpux::AddressType baseAlignment) {
    if (_optimizeFragmentation) {
        // sort to minimize fragmentation
        auto sortedBuffers = sortUsedBuffers(buffersToAllocate);
        // are resources available and can be allocated
        return _scan.canAllocWithReservedSpace(sortedBuffers, reserve, baseAlignment);
    }
    return _scan.canAllocWithReservedSpace(buffersToAllocate, reserve, baseAlignment);
}

vpux::AddressType FeasibleMemoryScheduler::sortAndAllocateBuffersWithReservedSize(
        mlir::DenseSet<mlir::Value>& buffersToAllocate, vpux::AddressType reserve, vpux::AddressType baseAlignment) {
    _log.nest().trace("Allocate memory for loop buffers");

    vpux::AddressType reservedBegin = 0;
    if (_optimizeFragmentation) {
        auto usedBuffers = sortUsedBuffers(buffersToAllocate);
        reservedBegin =
                _scan.allocWithReservedSpace(usedBuffers, reserve, baseAlignment, false, Partitioner::Direction::Up);
    } else {
        reservedBegin = _scan.allocWithReservedSpace(buffersToAllocate, reserve, baseAlignment, false,
                                                     Partitioner::Direction::Up);
    }

    VPUX_THROW_UNLESS(reservedBegin != std::numeric_limits<AddressType>::max(),
                      "Failed to statically allocate '{0}' memory", _memKind);
    return reservedBegin;
}
