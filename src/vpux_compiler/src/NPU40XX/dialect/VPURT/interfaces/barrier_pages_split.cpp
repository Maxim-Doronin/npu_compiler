//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"

#include <llvm/ADT/SetOperations.h>

using namespace vpux;

vpux::VPURT::BarrierPagesSplitHandler::BarrierPagesSplitHandler(BarrierInfo& barrierInfo, size_t numPhysBarriers,
                                                                Logger log)
        : _barrierInfo(barrierInfo), _barrierFifoDepth(BARRIER_FIFO_SIZE), _log(log) {
    VPUX_THROW_UNLESS(numPhysBarriers % 2 == 0, "Number of physical barriers must be even, numPhysBarriers - {0}",
                      numPhysBarriers);
    _pageSize = numPhysBarriers / 2;
}

// Below constructor is meant to be used only for unit testing purpose
vpux::VPURT::BarrierPagesSplitHandler::BarrierPagesSplitHandler(
        BarrierInfoTest& barrierInfoTest, std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueTypeMap,
        size_t pageSize, size_t _barrierFifoDepth, const SmallVector<size_t>& shvTasksWithDpu, Logger log)
        : _barrierInfo(barrierInfoTest),
          _pageSize(pageSize),
          _taskQueueTypeMap(taskQueueTypeMap),
          _barrierFifoDepth(_barrierFifoDepth),
          _log(log) {
    _pageCount = _barrierInfo.getNumOfBarrierOps() / _pageSize;
    if (_barrierInfo.getNumOfBarrierOps() % _pageSize) {
        _pageCount++;
    }
    _startBarrierIndex = 0;

    initializeTaskToPageAssignment();
    initializeBoundaryTasksData();

    // Initialize data for SHV tasks with DPU
    for (auto shvTaskInd : shvTasksWithDpu) {
        auto shvQueueIt = llvm::find_if(_taskQueueTypeMap, [&](const auto& item) {
            return llvm::find(item.second, shvTaskInd) != item.second.end();
        });
        VPUX_THROW_WHEN(shvQueueIt == _taskQueueTypeMap.end(), "Can not find task {0} in task queue map", shvTaskInd);

        _shvTasksWithDpuPerTile[shvQueueIt->first.id].push_back(shvTaskInd);
    }
}

void vpux::VPURT::BarrierPagesSplitHandler::reconfigureBarrierFifoDepth(size_t barrierFifoDepth) {
    _barrierFifoDepth = barrierFifoDepth;
}

// Configure the barrier page split handler for assignment of barriers and tasks to pages
void vpux::VPURT::BarrierPagesSplitHandler::initializeForAssignment(mlir::func::FuncOp func) {
    _taskQueueTypeMap = VPURT::getTaskOpQueues(func, _barrierInfo);
    _pageCount = (_barrierInfo.getNumOfBarrierOps() + _pageSize - 1) / _pageSize;
    initializeTaskToPageAssignment();
}

// Configure the barrier page split handler for legalization of the schedule for split into pages
void vpux::VPURT::BarrierPagesSplitHandler::initializeForLegalization() {
    mlir::DenseSet<vpux::VPU::ExecutorKind> executorKind = {
            VPU::ExecutorKind::DPU,
            VPU::ExecutorKind::DMA_NN,
            VPU::ExecutorKind::SHAVE_ACT,
    };
    _barrierInfo.initializeTaskQueueTypeMap(executorKind);
    _barrierInfo.buildTaskQueueTypeMap();

    _readPageAssignmentFromIr = true;

    // Get number of pages based on information in IR. Read page assignment from last barrier in IR
    auto lastBarOp = _barrierInfo.getBarrierOpAtIndex(_barrierInfo.getNumOfBarrierOps() - 1);
    auto lastPageOpt = lastBarOp.getWlmPage();
    VPUX_THROW_UNLESS(lastPageOpt.has_value(), "Barrier {0} does not have page assignment", lastBarOp);
    _pageCount = lastPageOpt.value() + 1;

    readTaskPageAssignmentFromIr();
    readBarrierPageAssignmentFromIr();

    initializeBoundaryTasksData();
}

void vpux::VPURT::BarrierPagesSplitHandler::findShvTasksWithDpu() {
    for (auto& [queueType, taskVec] : _taskQueueTypeMap) {
        if (queueType.type != VPU::ExecutorKind::SHAVE_ACT) {
            continue;
        }

        for (auto taskInd : taskVec) {
            if (isDpuShaveKernelType(_barrierInfo.getTaskOpAtIndex(taskInd))) {
                _shvTasksWithDpuPerTile[queueType.id].push_back(taskInd);
            }
        }
    }
    // Support for DPUs from SHV is not yet fully enabled - E#170833
    VPUX_THROW_UNLESS(_shvTasksWithDpuPerTile.empty(), "Full WLM does not yet support SHV tasks which submit DPUs");
}

// Configure the barrier page split handler for finding enqueue DMA data
void vpux::VPURT::BarrierPagesSplitHandler::initializeForEnqueue(mlir::func::FuncOp func) {
    initializeForLegalization();
    _taskQueueTypeMap = VPURT::getTaskOpQueues(func, _barrierInfo);
    findShvTasksWithDpu();
    initPrevPhysBarrierData(func);

    // Locate start barrier
    func.walk([&](VPURT::ConfigureBarrierOp barOp) {
        if (barOp.getIsStartBarrier()) {
            _startBarrierIndex = _barrierInfo.getIndex(barOp);
            return mlir::WalkResult::interrupt();
        }

        return mlir::WalkResult::advance();
    });
}

void vpux::VPURT::BarrierPagesSplitHandler::readBarrierPageAssignmentFromIr() {
    _barrierToPageAssignment.resize(_barrierInfo.getNumOfBarrierOps());

    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto barOp = _barrierInfo.getBarrierOpAtIndex(barInd);
        auto pageOpt = barOp.getWlmPage();
        VPUX_THROW_UNLESS(pageOpt.has_value(), "Barrier {0} does not have page assignment", barInd);
        _barrierToPageAssignment[barInd] = pageOpt.value();
    }
}

size_t vpux::VPURT::BarrierPagesSplitHandler::getBarrierPage(size_t barInd) {
    if (_readPageAssignmentFromIr) {
        VPUX_THROW_UNLESS(barInd < _barrierToPageAssignment.size(), "Barrier index {0} out of range {1}", barInd,
                          _barrierToPageAssignment.size());
        return _barrierToPageAssignment[barInd];
    }
    return static_cast<size_t>(barInd / _pageSize);
}

// Calculate in which page each task starts. Configure this based on task wait barrier and
// previous task (on same FIFO) page assignment.
// taskPage = max(waitBarriersPage, prevTaskPage)
void vpux::VPURT::BarrierPagesSplitHandler::initializeTaskToPageAssignment() {
    _taskPageAssignment.resize(_barrierInfo.getNumOfTasks());
    _log.trace("Initializing task to page assignment");

    for (auto& [_, taskVec] : _taskQueueTypeMap) {
        for (size_t i = 0; i < taskVec.size(); i++) {
            auto taskInd = taskVec[i];

            size_t taskPage = 0;

            // Get page assignment of wait barriers
            auto waitBars = _barrierInfo.getWaitBarriers(taskInd);

            if (!waitBars.empty()) {
                taskPage = getBarrierPage(*std::max_element(waitBars.begin(), waitBars.end()));
            }

            if (i > 0) {
                // Task page assignment cannot be smaller then previous task on same FIFO
                auto prevTaskInd = taskVec[i - 1];
                taskPage = std::max(taskPage, _taskPageAssignment[prevTaskInd]);
            }

            _taskPageAssignment[taskInd] = taskPage;
            _log.nest().trace("Task {0}: page {1}", taskInd, taskPage);
        }
    }
}

// Update boundary task data for provided task
void vpux::VPURT::BarrierPagesSplitHandler::updateBoundaryTasksDataForTask(size_t taskInd) {
    auto taskPage = _taskPageAssignment[taskInd];
    auto queueType = _barrierInfo.getTaskQueueType(taskInd);
    auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);
    auto waitBars = _barrierInfo.getWaitBarriers(taskInd);

    if (waitBars.empty() && updateBars.empty()) {
        // If there are no wait or update barriers, then no need to consider this task as
        // boundary task as such task has no tight affiliation to any page
        return;
    }

    // Check if there was already a boundary task of this type for this page identified
    if (_firstAndLastBoundaryTaskForEachPagePerFifo[taskPage].find(queueType) ==
        _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage].end()) {
        // If not check if all update barriers belong to the same page as the task.
        // If all update barriers are within taskPage, then this is NOT a boundary task
        if (llvm::all_of(updateBars, [&](size_t barInd) {
                return getBarrierPage(barInd) == taskPage;
            })) {
            return;
        }

        // If there is an update barrier from next page then this is a first boundary
        // task on this HW FIFO
        _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage][queueType] = std::make_pair(taskInd, taskInd);
    }

    if (taskInd < _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage][queueType].second) {
        return;
    }

    // If there is already a boundary task of this type for this page, then update the data
    // so that after traversing all tasks there is also information about the last boundary task
    _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage][queueType].second = taskInd;

    _log.nest().trace("Task {0} is boundary task for page {1}, queue type {2}:{3}", taskInd, taskPage,
                      stringifyEnum(queueType.type).data(), queueType.id);
}

// Make sure last boundary task on each HW FIFO updates barrier from next page.
// Tasks which are on the HW FIFO after a boundary task but they themselve do not have any
// update barrier from next page should have this legalized as rest of code expects this
// based on agreed definition of boundary task
void vpux::VPURT::BarrierPagesSplitHandler::enforceBoundaryTaskHasUpdateBarrier(size_t pageInd) {
    for (auto boundaryTask : getLastBoundaryTasksForPage(pageInd)) {
        auto taskUpdateBarsVec = to_small_vector(_barrierInfo.getUpdateBarriers(boundaryTask));

        // For analysis do not account for barriers from current page. Boundary task on PageN
        // needs to have update barriers from next page
        taskUpdateBarsVec.erase(llvm::remove_if(taskUpdateBarsVec,
                                                [&](auto barInd) {
                                                    return getBarrierPage(barInd) == pageInd;
                                                }),
                                taskUpdateBarsVec.end());
        if (!taskUpdateBarsVec.empty()) {
            // If task updates barriers from next page then no need to add any additional update barrier
            continue;
        }

        // If there are no update barriers of this page then need to search for some other update
        // barrier that can be used and which boundary task can update
        _log.nest().trace("Boundary task {0} from page {1} does not have update barrier on next pages", boundaryTask,
                          pageInd);

        // Check for update barrier of some previous task on the same FIFO
        auto currentTaskOpt = _barrierInfo.getPrevTaskOnSameQueue(boundaryTask);
        while (taskUpdateBarsVec.empty() && currentTaskOpt.has_value() &&
               _taskPageAssignment[currentTaskOpt.value()] == pageInd) {
            taskUpdateBarsVec = to_small_vector(_barrierInfo.getUpdateBarriers(currentTaskOpt.value()));

            taskUpdateBarsVec.erase(llvm::remove_if(taskUpdateBarsVec,
                                                    [&](auto barInd) {
                                                        return getBarrierPage(barInd) == pageInd;
                                                    }),
                                    taskUpdateBarsVec.end());

            currentTaskOpt = _barrierInfo.getPrevTaskOnSameQueue(currentTaskOpt.value());
        }

        VPUX_THROW_UNLESS(!taskUpdateBarsVec.empty(),
                          "Boundary task {0} from page {1} has no update barriers on next pages", boundaryTask,
                          pageInd);

        // If task updates multiple barriers, pick only one with smallest index and
        // update barrier dependencies so that boundary also updates this barrier
        // No need to use more barriers as 1 barrier is enough to know that boundary task has finished
        auto newUpdateBar = *std::min_element(taskUpdateBarsVec.begin(), taskUpdateBarsVec.end());

        _barrierInfo.addProducer(newUpdateBar, boundaryTask);
        _log.nest().trace("Add producer {0} to barrier {1}", boundaryTask, newUpdateBar);
    }
}

// Check all tasks and identify boundary tasks.
// A boundary task is one where at least one update barrier belongs to a page
// that is greater than taskPage (indicating cross-page dependency).
// Boundary tasks are later used for legalization purposes.
void vpux::VPURT::BarrierPagesSplitHandler::initializeBoundaryTasksData() {
    _firstAndLastBoundaryTaskForEachPagePerFifo.resize(_pageCount);
    _firstAndLastTaskPerPage.resize(_pageCount);
    _log.trace("Initializing boundary tasks data");

    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto taskPage = _taskPageAssignment[taskInd];

        if (!_firstAndLastTaskPerPage[taskPage].has_value()) {
            _firstAndLastTaskPerPage[taskPage] = std::make_pair(taskInd, taskInd);
        } else {
            _firstAndLastTaskPerPage[taskPage].value().second = taskInd;
        }

        updateBoundaryTasksDataForTask(taskInd);
    }

    for (size_t pageInd = 0; pageInd < _pageCount - 1; pageInd++) {
        auto pageBoundaryTasks = getFirstAndLastBoundaryTasksForPage(pageInd);
        if (pageBoundaryTasks.empty()) {
            _log.trace("No boundary tasks for page {0}", pageInd);
            // If in rare case there is no boundary task for page - task that starts at PageN and updates
            // barrier from PageN+1 then need to create one artificially. Pick lates task in PageN and find
            // some wait barrier from PageN+1 and create dependency.
            auto lastTaskOnPage = _firstAndLastTaskPerPage[pageInd].value().second;
            std::optional<size_t> nextPageBarrier;
            auto nextTask = lastTaskOnPage + 1;
            while (nextTask < _barrierInfo.getNumOfTasks() && _taskPageAssignment[nextTask] <= pageInd + 1 &&
                   !nextPageBarrier.has_value()) {
                auto nextTaskWaitBars = _barrierInfo.getWaitBarriers(nextTask);
                if (!nextTaskWaitBars.empty()) {
                    nextPageBarrier = *std::min_element(nextTaskWaitBars.begin(), nextTaskWaitBars.end());
                }

                nextTask++;
            }
            VPUX_THROW_UNLESS(nextPageBarrier.has_value(), "No next page barrier found for page {0} last task {1}",
                              pageInd, lastTaskOnPage);

            _barrierInfo.addProducer(nextPageBarrier.value(), lastTaskOnPage);
            _log.nest(2).trace("Add producer {0}(page {1}) to barrier {2}(page {3})", lastTaskOnPage, pageInd,
                               nextPageBarrier.value(), getBarrierPage(nextPageBarrier.value()));

            _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd][_barrierInfo.getTaskQueueType(lastTaskOnPage)] =
                    std::make_pair(lastTaskOnPage, lastTaskOnPage);
            pageBoundaryTasks.push_back(lastTaskOnPage);
        }

        _log.trace("Page {0} boundary tasks: {1}", pageInd, pageBoundaryTasks);
        enforceBoundaryTaskHasUpdateBarrier(pageInd);
    }
}

// Update boundary task data for provided page
void vpux::VPURT::BarrierPagesSplitHandler::updateBoundaryTasksDataForPage(size_t pageInd) {
    _log.trace("Update boundary tasks data for page {0}", pageInd);
    VPUX_THROW_WHEN(pageInd >= _pageCount, "Page index {0} out of range {1}", pageInd, _pageCount);

    // No need to update data for last page
    if (pageInd == _pageCount - 1) {
        return;
    }

    VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(), "No boundary tasks set for page {0}",
                    pageInd);
    VPUX_THROW_WHEN(!_firstAndLastTaskPerPage[pageInd].has_value(), "No first and last task set for page {0}", pageInd);

    _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].clear();
    for (size_t taskInd = _firstAndLastTaskPerPage[pageInd].value().first;
         taskInd <= _firstAndLastTaskPerPage[pageInd].value().second; taskInd++) {
        updateBoundaryTasksDataForTask(taskInd);
    }

    _log.trace("Page {0} boundary tasks: {1}", pageInd, getFirstAndLastBoundaryTasksForPage(pageInd));
    enforceBoundaryTaskHasUpdateBarrier(pageInd);
}

// For given page get first and last boundary tasks for each HW FIFO
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getFirstAndLastBoundaryTasksForPage(size_t pageInd) {
    auto boundaryTasks = getFirstBoundaryTasksForPage(pageInd);
    auto lastBoundaryTasks = getLastBoundaryTasksForPage(pageInd);
    boundaryTasks.insert(boundaryTasks.end(), lastBoundaryTasks.begin(), lastBoundaryTasks.end());

    llvm::sort(boundaryTasks);
    boundaryTasks.erase(std::unique(boundaryTasks.begin(), boundaryTasks.end()), boundaryTasks.end());
    return boundaryTasks;
}

// For given page get first boundary tasks for each HW FIFO
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getFirstBoundaryTasksForPage(size_t pageInd) {
    VPUX_THROW_UNLESS(pageInd < _firstAndLastBoundaryTaskForEachPagePerFifo.size(),
                      "Page index {0} not within limit {1}", pageInd,
                      _firstAndLastBoundaryTaskForEachPagePerFifo.size() - 1);

    SmallVector<size_t> boundaryTasks;

    for (auto& [_, firstLastTaskIndPair] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd]) {
        boundaryTasks.push_back(firstLastTaskIndPair.first);
    }

    llvm::sort(boundaryTasks);
    return boundaryTasks;
}

// For given page get last boundary tasks for each HW FIFO
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getLastBoundaryTasksForPage(size_t pageInd) {
    VPUX_THROW_UNLESS(pageInd < _firstAndLastBoundaryTaskForEachPagePerFifo.size(),
                      "Page index {0} not within limit {1}", pageInd,
                      _firstAndLastBoundaryTaskForEachPagePerFifo.size() - 1);

    SmallVector<size_t> boundaryTasks;

    for (auto& [_, firstLastTaskIndPair] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd]) {
        boundaryTasks.push_back(firstLastTaskIndPair.second);
    }

    llvm::sort(boundaryTasks);
    return boundaryTasks;
}

// Set barrier page assignment to barrier ops in IR
// This information is needed by physical barrier assignment pass or for later
// passes for logging and debug purpose
void vpux::VPURT::BarrierPagesSplitHandler::assignPagesToBarriersInIr() {
    // Assign pages to barriers
    _log.trace("Assigning pages to barriers");
    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto barOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(_barrierInfo.getBarrierOpAtIndex(barInd));
        auto barPage = getBarrierPage(barInd);
        barOp.setWlmPage(barPage);
        _log.nest().trace("Barrier {0} assigned to page {1}", barInd, barPage);
    }
}

// Set barrier page assignment to task ops in IR
// This information is used by page split passes
void vpux::VPURT::BarrierPagesSplitHandler::assignPagesToTasksInIr() {
    _log.trace("Assigning pages to tasks");
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto taskOp = _barrierInfo.getTaskOpAtIndex(taskInd);
        taskOp.setWlmPage(_taskPageAssignment[taskInd]);
    }
}

// Read barrier page assignment from IR. After first WLM page split runs, next passes
// are expected to read this data from IR
void vpux::VPURT::BarrierPagesSplitHandler::readTaskPageAssignmentFromIr() {
    // Assign pages to barriers
    _log.trace("Retrieve task to pages assignment from IR");
    _taskPageAssignment.resize(_barrierInfo.getNumOfTasks());
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto taskOp = _barrierInfo.getTaskOpAtIndex(taskInd);
        VPUX_THROW_UNLESS(taskOp.getWlmPageAttr() != nullptr,
                          "Get: attribute '{0}' was not set for '{1}' operation at '{2}'", VPURT::wlmPageAttrName,
                          taskOp->getName(), taskOp->getLoc());
        _taskPageAssignment[taskInd] = checked_cast<uint32_t>(taskOp.getWlmPage().value());
    }
}

// Check if given task has illegal dependencies from page split point of view
// If task is in pageN then:
// - if wait barrier is in earlier page (pageX, and X < N)
// - if update barrier page is later then next page (PageX, and X > N + 1)
// then task is considered to have too long dependency and this needs to be legalized
bool vpux::VPURT::BarrierPagesSplitHandler::isTaskWithNonAdjacentPageDependency(size_t taskInd) {
    auto taskPage = _taskPageAssignment[taskInd];

    // Check if any wait barrier is smaller than task page
    auto waitBars = _barrierInfo.getWaitBarriers(taskInd);
    if (llvm::any_of(waitBars, [&](const auto bar) {
            return getBarrierPage(bar) < taskPage;
        })) {
        return true;
    }

    // Check if any update barrier is greater than task page by more than 1
    auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);
    if (llvm::any_of(updateBars, [&](const auto bar) {
            return getBarrierPage(bar) > taskPage + 1;
        })) {
        return true;
    }

    return false;
}
// Check for each task if their barrier dependencies are not too long. If any task has
// illegal dependencies then split is not valid and needs to be legalized
bool vpux::VPURT::BarrierPagesSplitHandler::areNoDepsGoingBeyondNeighborPage() {
    _log.trace("Checking if no deps going beyond neighbor page");
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        if (isTaskWithNonAdjacentPageDependency(taskInd)) {
            _log.trace("Task {0} has deps going beyond neighbor page", taskInd);
            return false;
        }
    }
    _log.trace("All deps are within neighbor pages");
    return true;
}

// Get all tasks which have illegal - too long dependencies. These tasks will be legalized
// so that barrier dependencies satisfy barrier pages split constraint - task can use barriers
// only from adjacent pages
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getTasksWithNonAdjacentPageDependencyToLegalize() {
    _log.trace("Getting tasks with long bar deps to legalize");
    SmallVector<size_t> tasksWithDepsToLegalize;
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        if (isTaskWithNonAdjacentPageDependency(taskInd)) {
            tasksWithDepsToLegalize.push_back(taskInd);
        }
    }
    _log.trace("Number of tasks with long bar deps: {0}", tasksWithDepsToLegalize.size());
    return tasksWithDepsToLegalize;
}

// Legalize long dependency on wait barrier for a given task. If task on PageN
// has wait barrier on PageX, where X < N, then this dependency needs to be legalized
void vpux::VPURT::BarrierPagesSplitHandler::legalizeWaitBarrierDependency(size_t taskInd,
                                                                          VPURT::TaskQueueType& taskQueueType,
                                                                          size_t barInd) {
    size_t taskPage = _taskPageAssignment[taskInd];

    auto barPage = getBarrierPage(barInd);
    VPUX_THROW_WHEN(barPage > taskPage, "Task {0} from page {1} waits on barrier {2} from page {3}", taskInd, taskPage,
                    barInd, barPage);
    if (taskPage == barPage) {
        // Barrier is within accepted range
        // No need to legalize
        return;
    }
    _log.trace("Wait bar {0}(page {1}) needs to be legalized", barInd, barPage);

    // Find wait barrier of previous tasks on same FIFO. This barrier will be used for legalization
    // purpose as existing wait barrier will be disconnected from this task
    std::optional<size_t> waitBarOnSamePage = std::nullopt;
    std::optional<size_t> taskWithWaitBarOnSamePageOpt = taskInd;
    do {
        auto taskWaitBars = _barrierInfo.getWaitBarriers(taskWithWaitBarOnSamePageOpt.value());
        for (auto taskWaitBar : taskWaitBars) {
            if (getBarrierPage(taskWaitBar) == taskPage) {
                waitBarOnSamePage = taskWaitBar;
                break;
            }
        }
        if (waitBarOnSamePage.has_value()) {
            break;
        }

        taskWithWaitBarOnSamePageOpt =
                _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(taskWithWaitBarOnSamePageOpt.value());
    } while (!waitBarOnSamePage.has_value() && taskWithWaitBarOnSamePageOpt.has_value());

    VPUX_THROW_WHEN(!waitBarOnSamePage.has_value(), "No wait bar on page {0} for task {1}(page {2})", taskPage, taskInd,
                    taskPage);

    auto waitBarOnSamePageInd = waitBarOnSamePage.value();
    auto taskWithWaitBarOnSamePage = taskWithWaitBarOnSamePageOpt.value();
    _log.trace("Prev task on same FIFO Wait bar {0}(page {1})", waitBarOnSamePageInd,
               getBarrierPage(waitBarOnSamePageInd));

    // TODO: E#160461 Some improvement may be done by having some heuristic on which boundary
    // task should be used. Maybe take into account timing information or queue type?

    // Get producers of barInd to update boundaryTaskWaitBarInd
    auto barProdTasks = _barrierInfo.getBarrierProducers(barInd);
    for (auto barProdTask : barProdTasks) {
        auto barProdTaskPage = _taskPageAssignment[barProdTask];
        _log.nest().trace("Bar {0} producer: {1}(page {2})", barInd, barProdTask, barProdTaskPage);
        auto barProdQueueType = _barrierInfo.getTaskQueueType(barProdTask);

        if (taskQueueType == barProdQueueType) {
            _log.nest().trace("Task and producer are on the same FIFO {0}:{1}. No need to insert dependency "
                              "from barProdTask",
                              stringifyEnum(taskQueueType.type).data(), taskQueueType.id);
            continue;
        }

        if (taskPage - barPage == 1 && barProdTaskPage == barPage) {
            // Task and producer are on previous page. Connect task directly to closest wait barrier
            // of task that is to be legalized
            //
            // Example:
            // Before:
            // Task(PageN-1) -> BarToLegalize(PageN-1) ------------------|
            //                  Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //
            // After:
            // Task(PageN-1) -> BarToLegalize(PageN-1)
            //          |-----> Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)

            // Connect barProdTask to identified barrier but only if such dependency does not exist
            if (!isDepFromTaskAToTaskB(barProdTask, taskWithWaitBarOnSamePage)) {
                _barrierInfo.addProducer(waitBarOnSamePageInd, barProdTask);
                _log.nest(2).trace("Add producer {0}(page {1}) to barrier {2}(page {3}) which is wait barrier of "
                                   "task {4}(page {5}) on same FIFO as task {6}(page {7})",
                                   barProdTask, barProdTaskPage, waitBarOnSamePageInd, taskPage,
                                   taskWithWaitBarOnSamePage, _taskPageAssignment[taskWithWaitBarOnSamePage], taskInd,
                                   taskPage);
            }

        } else {
            // Barrier producer task is on earlier page than wait barrier that is to be legalized
            // or wait barrier is on earlier page than previous page where task is
            // Example 1:
            //
            //   Before:
            //   Task(PageN-2) -> BarToLegalize(PageN-1) ----------------------------|
            //                                        Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //
            //   After:
            //   Task(PageN-2) -> BarToLegalize(PageN-1)
            //       |---------> BoundaryTask(N-1) -> Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //
            // Example 2:
            //
            //   Before:
            //   Task(PageN-2) -> BarToLegalize(PageN-2) ----------------------------|
            //                                        Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //
            //   After:
            //   Task(PageN-2) -> BarToLegalize(PageN-2)
            //       |---------> BoundaryTask(N-1) -> Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //

            // TODO: E#160461 Analyze if this could be improved by picking different boundary task
            auto pageBoundaryTasks = getLastBoundaryTasksForPage(barProdTaskPage + 1);
            VPUX_THROW_WHEN(pageBoundaryTasks.empty(), "No boundary tasks set for page {0}", barProdTaskPage + 1);
            auto pageBoundaryTask = pageBoundaryTasks[0];
            _log.nest(2).trace("Page boundary task: {0}(page {1})", pageBoundaryTask,
                               _taskPageAssignment[pageBoundaryTask]);
            auto pageBoundaryTasksOnSameFifoIt =
                    _firstAndLastBoundaryTaskForEachPagePerFifo[barProdTaskPage + 1].find(barProdQueueType);
            if (pageBoundaryTasksOnSameFifoIt !=
                _firstAndLastBoundaryTaskForEachPagePerFifo[barProdTaskPage + 1].end()) {
                // There is a boundary task on same FIFO. No need to insert dependency from barProdTask
                pageBoundaryTask = pageBoundaryTasksOnSameFifoIt->second.second;
                _log.nest(2).trace("Page boundary task exists on same HW FIFO. No need to insert dependency "
                                   "from barProdTask. Use {0}(page {1}) as boundary task",
                                   pageBoundaryTask, _taskPageAssignment[pageBoundaryTask]);
            } else {
                auto pageBoundaryTaskWaitBars = _barrierInfo.getWaitBarriers(pageBoundaryTask);
                auto pageBoundaryTaskWithWaitBar = pageBoundaryTask;
                if (pageBoundaryTaskWaitBars.empty()) {
                    auto prevTaskOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(pageBoundaryTask);
                    if (prevTaskOpt.has_value()) {
                        pageBoundaryTaskWaitBars = _barrierInfo.getWaitBarriers(prevTaskOpt.value());
                        pageBoundaryTaskWithWaitBar = prevTaskOpt.value();
                    }
                }

                auto boundaryTaskWaitBarInd = *pageBoundaryTaskWaitBars.begin();
                _log.nest(2).trace("Boundary task {0} closest wait bar: {1}", pageBoundaryTask, boundaryTaskWaitBarInd);

                // Add dependency from task producing into barrier to be legalized to a wait barrier
                // of boundary task. Check if such dependency already exists
                if (!isDepFromTaskAToTaskB(barProdTask, pageBoundaryTaskWithWaitBar)) {
                    _barrierInfo.addProducer(boundaryTaskWaitBarInd, barProdTask);

                    _log.nest(2).trace("Add producer {0}(page {1}) to barrier {2}(page {3})", barProdTask,
                                       _taskPageAssignment[barProdTask], boundaryTaskWaitBarInd,
                                       getBarrierPage(boundaryTaskWaitBarInd));
                }
            }
            // Add dependency from pageBoundaryTask to taskInd but only if this pageBoundaryTask is on previous
            // page and there is no such dependency in place
            if (_taskPageAssignment[pageBoundaryTask] + 1 == taskPage &&
                !isDepFromTaskAToTaskB(pageBoundaryTask, taskInd)) {
                _barrierInfo.addProducer(waitBarOnSamePageInd, pageBoundaryTask);
                _log.nest(2).trace("Add dependency from boundary task {0}(page {1}) to task wait barrier {2}(page {3})",
                                   pageBoundaryTask, _taskPageAssignment[pageBoundaryTask], waitBarOnSamePageInd,
                                   getBarrierPage(waitBarOnSamePageInd));
            }
        }
    }
    _barrierInfo.removeConsumer(barInd, taskInd);
    _log.trace("Remove consumer {0}(page {1}) from barrier {2}(page {3})", taskInd, taskPage, barInd, barPage);
}

// Legalize long dependency on update barrier for a given task. If task on PageN
// has update barrier on PageX, where X > N+1, then this dependency needs to be legalized
void vpux::VPURT::BarrierPagesSplitHandler::legalizeUpdateBarrierDependency(size_t taskInd,
                                                                            VPURT::TaskQueueType& taskQueueType,
                                                                            size_t barInd) {
    size_t taskPage = _taskPageAssignment[taskInd];

    auto barPage = getBarrierPage(barInd);
    VPUX_THROW_WHEN(barPage < taskPage, "Task {0} from page {1} updates barrier {2} from page {3}", taskInd, taskPage,
                    barInd, barPage);
    if (barPage - taskPage <= 1) {
        // Barrier is within accepted range
        // No need to legalize
        return;
    }
    _log.trace("Update bar {0}(page {1}) needs to be legalized", barInd, barPage);

    // Task on PageN produces into barrier from PageX where X > N + 1
    // Remove this dependency and make task update wait barrier of boundary task from PageN+1
    //
    // Example:
    // Before:
    // Task(PageN) --------------------------------------------> BarToLegalize(PageN+2)
    //
    // After:
    // Task(PageN)                                               BarToLegalize(PageN+2)
    //          |-----> Bar(PageN+1) -> BoundaryTask(PageN+1) ---->|

    // TODO: E#160461 Some improvement may be done by having some heuristic on which boundary
    // task should be used. Maybe take into account timing information or queue type?
    auto pageBoundaryTasks = getLastBoundaryTasksForPage(taskPage + 1);
    VPUX_THROW_WHEN(pageBoundaryTasks.empty(), "No boundary tasks set for page {0}", taskPage + 1);
    auto pageBoundaryTask = pageBoundaryTasks[0];

    auto pageBoundaryTasksOnSameFifoIt = _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage + 1].find(taskQueueType);
    if (pageBoundaryTasksOnSameFifoIt != _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage + 1].end()) {
        // There is a boundary task on same FIFO. No need to insert dependency from barProdTask
        // Before adding the dependency, check for existing control path will prevent from adding this unnecessary
        // dependency
        pageBoundaryTask = pageBoundaryTasksOnSameFifoIt->second.second;
    }

    std::optional<size_t> pageBoundaryTaskWithWaitBarOpt = pageBoundaryTask;
    std::optional<size_t> pageBoundaryTaskWaitBarIndOpt = std::nullopt;

    // Start searching for a wait barrier from taskPage + 1 that is the one in this page that blocks
    // boundary task in this page. It doesn't have to be a wait barrier directly attached to this task
    // as it can have no wait barriers. It might be some previous task on same FIFO
    _log.trace("Start check from page boundary task {0}(page {1})", pageBoundaryTaskWithWaitBarOpt.value(),
               taskPage + 1);
    do {
        auto pageBoundaryTaskWaitBars = _barrierInfo.getWaitBarriers(pageBoundaryTaskWithWaitBarOpt.value());
        for (auto taskWaitBar : pageBoundaryTaskWaitBars) {
            if (getBarrierPage(taskWaitBar) == taskPage + 1) {
                pageBoundaryTaskWaitBarIndOpt = taskWaitBar;
                break;
            }
        }
        if (pageBoundaryTaskWaitBarIndOpt.has_value()) {
            break;
        }

        pageBoundaryTaskWithWaitBarOpt =
                _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(pageBoundaryTaskWithWaitBarOpt.value());

    } while (pageBoundaryTaskWithWaitBarOpt.has_value());

    VPUX_THROW_WHEN(!pageBoundaryTaskWithWaitBarOpt.has_value(), "No boundary task with valid barrier on page {0}",
                    taskPage + 1);

    auto pageBoundaryTaskWithWaitBar = pageBoundaryTaskWithWaitBarOpt.value();
    auto boundaryTaskWaitBarInd = pageBoundaryTaskWaitBarIndOpt.value();

    _log.trace("Use boundary task {0}(page {1}) and its wait barrier {2}(page {3})", pageBoundaryTaskWithWaitBar,
               _taskPageAssignment[pageBoundaryTaskWithWaitBar], boundaryTaskWaitBarInd,
               getBarrierPage(boundaryTaskWaitBarInd));

    // Before adding taskInd to boundary task dep, check if such dependency does not already exist
    if (!isDepFromTaskAToTaskB(taskInd, pageBoundaryTaskWithWaitBar)) {
        _barrierInfo.addProducer(boundaryTaskWaitBarInd, taskInd);
        _log.trace("Add producer {0}(page {1}) to boundary task {2}(page {3}) wait barrier {4}(page {5})", taskInd,
                   taskPage, pageBoundaryTaskWithWaitBar, _taskPageAssignment[pageBoundaryTaskWithWaitBar],
                   boundaryTaskWaitBarInd, getBarrierPage(boundaryTaskWaitBarInd));
    }

    // Before adding a dependency from boundary task to barInd, check if such dependency does not already
    // exist by checking if there is a control path between boundary task and any producer of barInd
    if (barPage == _taskPageAssignment[pageBoundaryTask] + 1 && !isDepFromTaskToBarrier(pageBoundaryTask, barInd)) {
        _barrierInfo.addProducer(barInd, pageBoundaryTask);
        _log.trace("Add producer {0}(page {1}) to barrier {2}(page {3})", pageBoundaryTask,
                   _taskPageAssignment[pageBoundaryTask], barInd, barPage);
    }

    _barrierInfo.removeProducer(barInd, taskInd);
    _log.trace("Remove producer {0}(page {1}) from barrier {2}(page {3})", taskInd, taskPage, barInd, barPage);
}

// Legalize all too long dependencies for tasks identified by getTasksWithNonAdjacentPageDependencyToLegalize
// This is core legalization method and it will modify task-barrier dependencies so that task
// updates only barriers from its and next page
// This method does not create any new barriers
void vpux::VPURT::BarrierPagesSplitHandler::legalizeNonAdjacentPageDependencies(
        SmallVector<size_t>& tasksWithDepsToLegalize) {
    _log.trace("Legalizing long deps beyond next page");
    _log = _log.nest();
    for (auto taskInd : tasksWithDepsToLegalize) {
        size_t taskPage = _taskPageAssignment[taskInd];
        auto taskQueueType = _barrierInfo.getTaskQueueType(taskInd);
        _log.trace("Task {0}(page {1}) with long dep needs to be legalized", taskInd, taskPage);
        _log = _log.nest();

        // Check if there are any wait barriers in the task that represent long dep.
        // Wait barrier page needs to match task page assignment
        auto waitBars = _barrierInfo.getWaitBarriers(taskInd);

        _log.trace("Checking wait barriers");
        for (auto barInd : waitBars) {
            legalizeWaitBarrierDependency(taskInd, taskQueueType, barInd);
        }

        // Check if there are any update barriers in the task that represent long dep.
        // Update barrier page needs to match task page (PageN) or next page (PageN+1)
        auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);

        _log.trace("Checking update barriers");
        for (auto barInd : updateBars) {
            legalizeUpdateBarrierDependency(taskInd, taskQueueType, barInd);
        }
        _log = _log.unnest();
    }
    _log = _log.unnest();

    // Invalidate task control map after barrier deps have changed
    _blockIdxOfTaskControlMap = std::nullopt;
}

// Check if there is a dependency from taskA to taskB taking into account barrier dependency
// and HW FIFOs
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromTaskAToTaskB(size_t taskA, size_t taskB) {
    return _barrierInfo.isDepFromTaskAToTaskB(taskA, taskB, _taskControlMapAndOffset, _blockIdxOfTaskControlMap);
}

// Check if there is a dependency from task to barrier by checking if there is any dependency
// from this task to any producer of this barrier. If yes then there is a guarantee that task
// needs to execute before barrier is produced - there is topological dependency
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromTaskToBarrier(size_t taskInd, size_t barInd) {
    return _barrierInfo.isDepFromTaskToBarrier(taskInd, barInd, _taskControlMapAndOffset, _blockIdxOfTaskControlMap);
}

// Check if there is a dependency from barA to barB by checking if there is any dependency
// from barA consumer to barB producer
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromBarAToBarB(size_t barA, size_t barB) {
    return _barrierInfo.isDepFromBarAToBarB(barA, barB, _taskControlMapAndOffset, _blockIdxOfTaskControlMap);
}

// Check if boundary task of consecutive pages depend on each other
// Example:
//   PageN boundary tasks:   taskA, taskB
//   PageN+1 boundary tasks: taskC, taskD
// There needs to be dependency from taskA to taskC&D and taskB to taskC&D
// This way when any of PageN+1 boundary tasks executes it is guaranteed that
// all barriers from PageN have been consumed and they are configured for PageN+2 and
// boundary tasks of PageN+1 are first tasks which produce into barriers from PageN+1
bool vpux::VPURT::BarrierPagesSplitHandler::areBoundaryTasksFromNeighborPagesDependent() {
    _log.trace("Checking if boundary tasks from neighbor pages are dependent");
    if (_pageCount <= 2) {
        return true;
    }

    // Check deps from PageN to PageN+1 boundary tasks
    for (size_t pageInd = 0; pageInd < _pageCount - 2; pageInd++) {
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(),
                        "No boundary tasks set for page {0}", pageInd);
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd + 1].empty(),
                        "No boundary tasks set for page {0}", pageInd + 1);
        auto pageBoundaryTaskPerFifo = _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd];
        auto nextPageBoundaryTaskPerFifo = _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd + 1];

        for (auto [_, pageFirstLastBoundaryTask] : pageBoundaryTaskPerFifo) {
            for (auto [_, nextPageFirstLastBoundaryTask] : nextPageBoundaryTaskPerFifo) {
                auto task1 = pageFirstLastBoundaryTask.second;
                auto task2 = nextPageFirstLastBoundaryTask.first;
                if (!isDepFromTaskAToTaskB(task1, task2)) {
                    _log.trace("Boundary tasks {0} and {1} are not dependent", task1, task2);
                    return false;
                }
            }
        }
    }

    return true;
}

// Get all boundary task pairs from neighbor pages which are missing dependency in between
// If boundary task from PageN+1 misses dependency from boundary task from PageN then
// return such pair
SmallVector<std::pair<size_t, size_t>>
vpux::VPURT::BarrierPagesSplitHandler::getBoundaryTaskPairsMissingDepInBetween() {
    SmallVector<std::pair<size_t, size_t>> boundaryTaskPairsMissingDepInBetween;

    _log.trace("Getting boundary task pairs missing dep in between");
    if (_pageCount <= 2) {
        return boundaryTaskPairsMissingDepInBetween;
    }

    // Check deps from PageN to PageN+1 boundary tasks
    for (size_t pageInd = 0; pageInd < _pageCount - 2; pageInd++) {
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(),
                        "No boundary tasks set for page {0}", pageInd);
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd + 1].empty(),
                        "No boundary tasks set for page {0}", pageInd + 1);
        auto pageBoundaryTaskPerFifo = _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd];
        auto nextPageBoundaryTaskPerFifo = _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd + 1];

        for (auto [_, pageFirstLastBoundaryTask] : pageBoundaryTaskPerFifo) {
            for (auto [_, nextPageFirstLastBoundaryTask] : nextPageBoundaryTaskPerFifo) {
                auto task1 = pageFirstLastBoundaryTask.second;
                auto task2 = nextPageFirstLastBoundaryTask.first;
                if (isDepFromTaskAToTaskB(task1, task2)) {
                    continue;
                }
                boundaryTaskPairsMissingDepInBetween.push_back(std::make_pair(task1, task2));
            }
        }
    }

    _log.trace("Number of boundary task pairs missing dep in between: {0}",
               boundaryTaskPairsMissingDepInBetween.size());
    return boundaryTaskPairsMissingDepInBetween;
}

// Legalize missing dependencies in between of boundary tasks from neighbor pages
// This method is meant to work on result from getBoundaryTaskPairsMissingDepInBetween
void vpux::VPURT::BarrierPagesSplitHandler::legalizeDepsForBoundaryTasks(
        SmallVector<std::pair<size_t, size_t>>& boundaryTaskPairsMissingDepInBetween) {
    _log.trace("Legalizing deps for boundary tasks");
    for (auto [taskSrc, taskDst] : boundaryTaskPairsMissingDepInBetween) {
        auto waitBars = _barrierInfo.getWaitBarriers(taskDst);
        if (waitBars.empty()) {
            auto prevTaskOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(taskDst);
            if (prevTaskOpt.has_value()) {
                waitBars = _barrierInfo.getWaitBarriers(prevTaskOpt.value());
            }
        }

        VPUX_THROW_WHEN(waitBars.empty(), "Task {0} has no wait barrier to connect to", taskDst);
        auto waitBar = *waitBars.begin();

        // insert dependency from taskSrc to wait barrier of taskDst to create taskSrc->taskDst dependency
        _log.trace("Create dependency from task {0}(page {1}) to task {2}(page {3})", taskSrc,
                   _taskPageAssignment[taskSrc], taskDst, _taskPageAssignment[taskDst]);
        _log.trace("Add producer {0}(page {1}) to barrier {2}(page {3})", taskSrc, _taskPageAssignment[taskSrc],
                   waitBar, getBarrierPage(waitBar));
        _barrierInfo.addProducer(waitBar, taskSrc);
    }

    // Invalidate task control map after barrier deps have changed
    _blockIdxOfTaskControlMap = std::nullopt;
}

bool vpux::VPURT::BarrierPagesSplitHandler::isSplitToPagesValid() {
    _log.trace("Checking if split to pages is valid");
    if (!areNoDepsGoingBeyondNeighborPage()) {
        return false;
    }

    if (!areBoundaryTasksFromNeighborPagesDependent()) {
        return false;
    }

    return true;
}

// Based on barriers that will be used as wait barriers (pageStartBars) and update barriers (pageEndBars) of barrier DMA
// find a place in IR (index of operation) after which this DMA can be inserted
size_t vpux::VPURT::BarrierPagesSplitHandler::getInsertionPointForDmaProgrammingBarriers(
        const BarrierInfo::TaskSet& pageStartBars, const BarrierInfo::TaskSet& pageEndBars) {
    auto startPoint = _barrierInfo.getBarriersLatestProducer(pageStartBars);
    auto endPoint = _barrierInfo.getBarriersEarliestConsumer(pageEndBars);

    _log.trace("Find insertion point between {0} and {1}", startPoint, endPoint);

    if (startPoint < endPoint) {
        // Simple case where all pageStartBars producers are before pageEndBars consumers
        // and insertion point can be set to latest producer of pageStartBars
        _log.trace("Insert BarProgDMA after task {0}", startPoint);
        return startPoint;
    }

    _log.trace("Check start and end barrier usage before picking insertion point");

    // endPoint < startPoint
    // There can be a case where pageStartBars producers are after pageEndBars consumers based on index
    // in IR, for example this can happen when there are parallel branches and there are multiple ways
    // how tasks from unrelated branches are assigned indexes and placed in IR.
    // There are two solutions
    //  - insert after startPoint if any of pageStartBars are produced by DMA on the same FIFO as BarProgDMA
    //  - insert before endPoint if any of pageEndBars are consumed by DMA on the same FIFO as BarProgDMA

    VPURT::TaskQueueType barProgDmaQueueType;
    barProgDmaQueueType.type = VPU::ExecutorKind::DMA_NN;
    barProgDmaQueueType.id = getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR);

    bool isAnyPageStartBarsProducersOnSameFifo = false;
    for (auto barInd : pageStartBars) {
        auto barProdTasks = _barrierInfo.getBarrierProducers(barInd);
        for (auto barProdTask : barProdTasks) {
            // Interested only in range [endPoint, startPoint]
            if (barProdTask < endPoint) {
                continue;
            }
            if (_barrierInfo.getTaskQueueType(barProdTask) == barProgDmaQueueType) {
                _log.trace("Task {0} which produces barrier {1} is a DMA P:0 CH:DDR", barProdTask, barInd);
                isAnyPageStartBarsProducersOnSameFifo = true;
                break;
            }
        }
        if (isAnyPageStartBarsProducersOnSameFifo) {
            break;
        }
    }

    bool isAnyPageEndBarsConsumersOnSameFifo = false;
    for (auto barInd : pageEndBars) {
        auto barConsTasks = _barrierInfo.getBarrierConsumers(barInd);
        for (auto barConsTask : barConsTasks) {
            // Interested only in range [endPoint, startPoint]
            if (barConsTask > endPoint) {
                continue;
            }
            if (_barrierInfo.getTaskQueueType(barConsTask) == barProgDmaQueueType) {
                _log.trace("Task {0} which consumes barrier {1} is a DMA P:0 CH:DDR", barConsTask, barInd);
                isAnyPageEndBarsConsumersOnSameFifo = true;
                break;
            }
        }
        if (isAnyPageEndBarsConsumersOnSameFifo) {
            break;
        }
    }

    // Theoretically we should not run into below exception as pageStartBar dependency on pageEndBars
    // was checked during legalization so there should be a way to correctly insert BarProgDMA
    VPUX_THROW_WHEN(isAnyPageEndBarsConsumersOnSameFifo && isAnyPageStartBarsProducersOnSameFifo,
                    "Both pageStartBars and pageEndBars have producers/consumers on the same FIFO as BarProgDMA");

    if (isAnyPageEndBarsConsumersOnSameFifo) {
        _log.trace("Insert BarProgDMA before task {0}", endPoint);
        return endPoint - 1;
    }

    _log.trace("Insert BarProgDMA after task {0}", startPoint);

    return startPoint;
}

// In case page crosses control block boundary then creating a barrier DMA that will wait
// on barriers from one block and update barriers of other block would violate control block split.
// To prevent from that adjust start barriers or end barriers if page crosses control block boundary
// Treat control block sync task as a start or end point for barrier DMA legalization purpose
void vpux::VPURT::BarrierPagesSplitHandler::adjustPageStartAndEndPointsIfOnBlockBoundary(
        BarrierInfo::TaskSet& pageStartBars, BarrierInfo::TaskSet& pageStartTasks, BarrierInfo::TaskSet& pageEndBars,
        BarrierInfo::TaskSet& pageEndAllBars, BarrierInfo::TaskSet& pageEndTasks) {
    // Barrier programming DMA to be inserted after page boundary tasks so that it is the first task on the page
    auto initialInsertionPosition = *std::max_element(pageStartTasks.begin(), pageStartTasks.end());
    // Use initialInsertionPosition task + 1 because bar programming DMA is inserted after this task
    auto barProgDmaBlockInd = _barrierInfo.getControlGraphBlockIndex(initialInsertionPosition + 1);
    size_t pageEndBarBlockInd =
            _barrierInfo.getControlGraphBlockIndex(*std::min_element(pageEndTasks.begin(), pageEndTasks.end()));
    if (barProgDmaBlockInd < pageEndBarBlockInd) {
        // If page end is on another control block then treat control block sync task as page boundary task
        auto syncTaskOpt = _barrierInfo.getControlGraphSyncPoint(initialInsertionPosition);
        VPUX_THROW_UNLESS(syncTaskOpt.has_value(), "No control block sync task for task {0}", initialInsertionPosition);

        auto syncTaskWaitBars = _barrierInfo.getWaitBarriers(syncTaskOpt.value());

        if (syncTaskWaitBars.size() == 1 && pageStartBars == syncTaskWaitBars) {
            // If sync task is the only consumer of page start barriers then use it as start point
            // for legalization - Control block sync point will be page start task
            _log.trace("Control block boundary: {0} and {1}. Use sync task {2} as startpoint for legalization",
                       barProgDmaBlockInd, pageEndBarBlockInd, syncTaskOpt.value());

            auto syncTaskUpdateBars = _barrierInfo.getUpdateBarriers(syncTaskOpt.value());
            VPUX_THROW_UNLESS(!syncTaskUpdateBars.empty(), "Control block sync task {0} has no update barriers",
                              syncTaskOpt.value());
            auto newPageStartBar = *std::min_element(syncTaskUpdateBars.begin(), syncTaskUpdateBars.end());

            pageStartBars.clear();
            pageStartTasks.clear();
            pageStartBars.insert(newPageStartBar);
            pageStartTasks.insert(syncTaskOpt.value());
        } else {
            // Use control block sync point as page end task
            _log.trace("Control block boundary: {0} and {1}. Use sync task {2} as endpoint for legalization",
                       barProgDmaBlockInd, pageEndBarBlockInd, syncTaskOpt.value());

            VPUX_THROW_UNLESS(!syncTaskWaitBars.empty(), "Control block sync task {0} has no wait barriers",
                              syncTaskOpt.value());
            auto newPageEndBar = *std::max_element(syncTaskWaitBars.begin(), syncTaskWaitBars.end());
            pageEndBars.clear();
            pageEndAllBars.clear();
            pageEndTasks.clear();
            pageEndBars.insert(newPageEndBar);
            pageEndAllBars.insert(newPageEndBar);
            pageEndTasks.insert(syncTaskOpt.value());
        }
    }
}

BarrierInfo::TaskSet vpux::VPURT::BarrierPagesSplitHandler::getPageStartBarsDependingOnPageEndBars(
        BarrierInfo::TaskSet& pageStartBars, BarrierInfo::TaskSet& pageEndBars) {
    BarrierInfo::TaskSet pageStartBarsToLegalize;
    for (auto pageStartBar : pageStartBars) {
        for (auto pageEndBar : pageEndBars) {
            if (!isDepFromBarAToBarB(pageEndBar, pageStartBar)) {
                continue;
            }
            _log.nest().trace("There is dep from page end bar {0} to page start bar {1}", pageEndBar, pageStartBar);
            pageStartBarsToLegalize.insert(pageStartBar);
            break;
        }
    }
    return pageStartBarsToLegalize;
}

// Check if any pageStartBar depends on any pageEndBar. Such pageStartBar cannot be used as a starting point
// for a barrier programming DMA which will also produce into pageEndBars. This function will identify such barrier,
// remove it from pageStartBars set and add necessary dependency from boundary task to other, valid pageStartBar
void vpux::VPURT::BarrierPagesSplitHandler::legalizePageStartBarsDependingOnPageEndBars(
        BarrierInfo::TaskSet& pageStartTasks, BarrierInfo::TaskSet& pageStartBars,
        BarrierInfo::TaskSet& pageStartBarsToLegalize) {
    _log.trace("Legalize page start bars depending on page end bars");

    // Pick some other pageStartBar that will be used for legalization
    VPUX_THROW_UNLESS(!pageStartBars.empty(), "No page start bars to use for legalization");
    auto startBarToUseForLegalization = *std::max_element(pageStartBars.begin(), pageStartBars.end());

    for (auto pageStartBarToLegalize : pageStartBarsToLegalize) {
        // Legalize by identifying boundary tasks that were producers of pageStartBarToLegalize
        // and make them update other pageStartBar - startBarToUseForLegalization
        _log.trace("Legalizing page start bar {0} which cannot be used as wait barrier for DMA",
                   pageStartBarToLegalize);

        // Find all boundary tasks that is producers of pageStartBarToLegalize
        auto startBarProducersToBeLegalized = to_small_vector(_barrierInfo.getBarrierProducers(pageStartBarToLegalize));

        // Leave only producers which are boundary tasks
        startBarProducersToBeLegalized.erase(llvm::remove_if(startBarProducersToBeLegalized,
                                                             [&](auto taskInd) {
                                                                 return llvm::find(pageStartTasks, taskInd) ==
                                                                        pageStartTasks.end();
                                                             }),
                                             startBarProducersToBeLegalized.end());

        for (auto startBarProducerToBeLegalized : startBarProducersToBeLegalized) {
            _barrierInfo.addProducer(startBarToUseForLegalization, startBarProducerToBeLegalized);
            _log.trace("Add producer {0} to barrier {1}", startBarProducerToBeLegalized, startBarToUseForLegalization);
        }
    }
}

// Get boundary tasks (last on each HW FIFO) from pageInd-1 to pageInd and get barriers they update
// This information will be used as starting point for a page - mark the start
// of barrier DMA
void vpux::VPURT::BarrierPagesSplitHandler::getPageStartTasksAndBars(size_t pageInd,
                                                                     BarrierInfo::TaskSet& pageStartTasks,
                                                                     BarrierInfo::TaskSet& pageStartBars) {
    VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1].empty(),
                    "No boundary tasks set for page {0}", pageInd - 1);
    for (auto& [taskQueueType, firstLastTaskInd] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1]) {
        _log.trace("Get page start tasks and bars for queue {0}:{1}", stringifyEnum(taskQueueType.type).data(),
                   taskQueueType.id);
        auto pageStartTask = firstLastTaskInd.second;
        auto taskUpdateBarsVec = to_small_vector(_barrierInfo.getUpdateBarriers(pageStartTask));

        // Remove barriers that are not on this page - they can be from previous page
        taskUpdateBarsVec.erase(llvm::remove_if(taskUpdateBarsVec,
                                                [&](auto barInd) {
                                                    return getBarrierPage(barInd) != pageInd;
                                                }),
                                taskUpdateBarsVec.end());

        VPUX_THROW_UNLESS(!taskUpdateBarsVec.empty(), "Page start task {0} has no update barriers on page {1}",
                          pageStartTask, pageInd);

        // If task updates multiple barriers, pick only one with smallest index
        // No need to use more barriers as 1 barrier is enough to know that boundary task have finished
        auto startBar = *std::min_element(taskUpdateBarsVec.begin(), taskUpdateBarsVec.end());
        pageStartBars.insert(startBar);
        pageStartTasks.insert(pageStartTask);
        _log.nest().trace("Page start task {0}, start bar {1}", pageStartTask, startBar);
    }
}

// Get boundary tasks from (first on each HW FIFO) pageInd to pageInd+1 and get barriers they wait on
// Get also remaining wait barriers for other boundary tasks on each FIFO for this page
// This information will be used as end point for a page - mark the end
// of barrier DMA
void vpux::VPURT::BarrierPagesSplitHandler::getPageEndTasksAndBars(size_t pageInd, BarrierInfo::TaskSet& pageEndTasks,
                                                                   BarrierInfo::TaskSet& pageEndBars,
                                                                   BarrierInfo::TaskSet& pageEndAllBars) {
    VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(), "No boundary tasks set for page {0}",
                    pageInd);
    for (auto& [taskQueueType, firstLastTaskInd] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd]) {
        _log.trace("Get page end tasks and bars for queue {0}:{1}", stringifyEnum(taskQueueType.type).data(),
                   taskQueueType.id);
        auto pageEndTask = firstLastTaskInd.first;
        auto taskWaitBars = _barrierInfo.getWaitBarriers(pageEndTask);

        if (taskWaitBars.empty()) {
            auto taskWithWaitBarOnSamePageOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(pageEndTask);
            VPUX_THROW_UNLESS(taskWithWaitBarOnSamePageOpt.has_value(), "No wait barriers for task {0}(page {1})",
                              pageEndTask, _taskPageAssignment[pageEndTask]);

            pageEndTask = taskWithWaitBarOnSamePageOpt.value();
            taskWaitBars = _barrierInfo.getWaitBarriers(pageEndTask);
        }
        // If task wait on multiple barriers, pick only one with latest index
        // No need to use more barriers as 1 barrier is enough to block next boundary task
        auto endBar = *std::max_element(taskWaitBars.begin(), taskWaitBars.end());
        pageEndBars.insert(endBar);
        pageEndTasks.insert(pageEndTask);
        _log.nest().trace("Page end task {0}, end bar {1}", pageEndTask, endBar);

        // Get all wait barriers that are used by boundary tasks
        // Scan whole range of boundary task on each FIFO to get all wait barriers
        std::optional<size_t> currentTaskOpt = pageEndTask;
        auto lastTask = firstLastTaskInd.second;
        while (currentTaskOpt.has_value() && currentTaskOpt.value() <= lastTask) {
            auto taskWaitBars = _barrierInfo.getWaitBarriers(currentTaskOpt.value());
            if (!taskWaitBars.empty()) {
                _log.nest().trace("Page end task {0}, end bars {1}", currentTaskOpt.value(),
                                  to_small_vector(taskWaitBars));
                pageEndAllBars.insert(taskWaitBars.begin(), taskWaitBars.end());
            }
            currentTaskOpt = _barrierInfo.getNextTaskOnSameQueue(currentTaskOpt.value());
        }
    }
}

// Additional legalization is needed if compiler intends to prepare barrier programming DMAs
// Page split needs to be adjusted to create place in schedule with barriers that mark page start and page end
// and allow for insertion of a barrier programming DMA that will run during execution of PageN
// and is guaranteed to start after all barriers from PageN-1 are consumed and will finish before
// any task producing barrier to PageN+1 start.
// Pages which will be legalized for barrier DMAs depend on Barrier FIFO depth.
// Expectation is following in case Barrier FIFO depth = 4:
// -------------------------------------------------------------------------------------
// Barrier DMA executed at: |	Physical barriers programmed | Barrier pages programmed
// -------------------------------------------------------------------------------------
//   Bootstrap	            |  [0, N-1]     (all)	         | Page0-Page7
//   Page7	                |  [0, (N/2)-1] (1st half)	     | Page8/10/12/14
//   Page8	                |  [(N/2), N-1] (2nd half)	     | Page9/11/13/15
//   Page15	                |  [0, (N/2)-1] (1st half)	     | Page16/18/20/22
//   Page16	                |  [(N/2), N-1] (2nd half)	     | Page17/19/21/23
//
void vpux::VPURT::BarrierPagesSplitHandler::legalizeForDmaProgrammingBarriers() {
    size_t barrierFifoDepthInPages = 2 * _barrierFifoDepth;
    if (_pageCount <= barrierFifoDepthInPages) {
        // No need to do any legalization as all barriers will be configured at bootstrap
        _log.trace("No need to legalize for barrier programming DMAs due to small number of pages ({0}) compared to "
                   "barrier FIFO depth ({1})",
                   _pageCount, _barrierFifoDepth);
        return;
    }

    _barProgDmaPosVec.resize(_pageCount - 1);

    auto setBarProgDmaData = [&](size_t pageInd, const BarrierInfo::TaskSet& pageStartBars,
                                 const BarrierInfo::TaskSet& pageEndBars) {
        _barProgDmaPosVec[pageInd] = {true, SmallVector<size_t>(pageStartBars.begin(), pageStartBars.end()),
                                      SmallVector<size_t>(pageEndBars.begin(), pageEndBars.end()),
                                      getInsertionPointForDmaProgrammingBarriers(pageStartBars, pageEndBars)};

        llvm::sort(_barProgDmaPosVec[pageInd].waitBars);
        llvm::sort(_barProgDmaPosVec[pageInd].updateBars);
    };

    // pageInd refers to page that will contain a barrier programming DMA for pageInd+1
    // Initial pages are expected to be programmed at bootstrap
    for (size_t pageInd = barrierFifoDepthInPages - 1; pageInd < _pageCount - 1; pageInd++) {
        if (pageInd % barrierFifoDepthInPages != 0 && (pageInd + 1) % barrierFifoDepthInPages != 0) {
            // Barrier programming DMA is not needed in this page
            continue;
        }

        _log.trace("Legalizing for barrier programming DMA for page {0}", pageInd);
        _log = _log.nest();
        // Get boundary tasks to identify which barriers mark the start and end of page
        // Those barriers will be used as an indication for where the barrier DMA needs to be placed

        // Get boundary tasks from pageInd-1 to pageInd and get barriers they update
        // Those barriers will mark the start of DMA
        BarrierInfo::TaskSet pageStartTasks;
        BarrierInfo::TaskSet pageStartBars;
        getPageStartTasksAndBars(pageInd, pageStartTasks, pageStartBars);
        VPUX_THROW_WHEN(pageStartBars.empty(), "No start barriers for page {0} identified", pageInd);

        // Get boundary tasks (first on each HW FIFO) from pageInd and get barriers they wait on
        // Those barriers will mark the end of DMA
        // Get also remaining wait barriers for other boundary tasks on each FIFO for this page
        BarrierInfo::TaskSet pageEndTasks;
        BarrierInfo::TaskSet pageEndBars;
        BarrierInfo::TaskSet pageEndAllBars;
        getPageEndTasksAndBars(pageInd, pageEndTasks, pageEndBars, pageEndAllBars);
        VPUX_THROW_WHEN(pageEndBars.empty(), "No end barriers for page {0} identified", pageInd);

        // Adjust start barriers or end barriers if page crosses control block boundary
        // In such case treat control block sync task as a start or end point for barrier DMA
        // legalization purpose
        adjustPageStartAndEndPointsIfOnBlockBoundary(pageStartBars, pageStartTasks, pageEndBars, pageEndAllBars,
                                                     pageEndTasks);

        // Check if there are any common barriers between pageStartBars and pageEndBars
        auto commonStartEndBars = llvm::set_intersection(pageStartBars, pageEndAllBars);

        // Common barriers need to be legalized and related task dependencies updated
        // If there is already other start barrier not part of common barrier, make all common barriers
        // to be end barriers
        auto pageStartOnlyBars = llvm::set_difference(pageStartBars, commonStartEndBars);

        // Check if any pageStartBar depends on any pageEndBar. In that case it cannot be used
        // as wait barrier for barrier DMA
        BarrierInfo::TaskSet pageStartBarsToLegalize =
                getPageStartBarsDependingOnPageEndBars(pageStartOnlyBars, pageEndBars);

        llvm::set_subtract(pageStartOnlyBars, pageStartBarsToLegalize);

        // During legalization dependencies are modified and boundary tasks data may change
        // If tasks and barriers are from different pages, store this information so that
        // boundary data for this page can be updated and barrier programming DMA legalization
        // on next page can have up to date info about boundary tasks
        mlir::DenseSet<size_t> pagesWithPossibleBoundaryTasksChanged;
        auto barrierDepsChangedHandler = [&](size_t barInd, size_t taskInd) {
            auto barPage = getBarrierPage(barInd);
            auto taskPage = _taskPageAssignment[taskInd];
            if (barPage != taskPage) {
                pagesWithPossibleBoundaryTasksChanged.insert(taskPage);
                pagesWithPossibleBoundaryTasksChanged.insert(barPage);
            }
        };

        if (!pageStartOnlyBars.empty()) {
            legalizePageStartBarsDependingOnPageEndBars(pageStartTasks, pageStartOnlyBars, pageStartBarsToLegalize);

            // ----------------------------------------
            // Case 1: No common start and end barriers
            // ----------------------------------------
            if (commonStartEndBars.empty()) {
                _log.trace("No common start and end barriers");

                setBarProgDmaData(pageInd, pageStartOnlyBars, pageEndBars);
                _log = _log.unnest();
                continue;
            }

            _log.trace("Legalizing common start and end barriers - count {0}", commonStartEndBars.size());

            // ------------------------------------------------------------------
            // Case 2: There are barriers which can be treated as start barriers
            // ------------------------------------------------------------------

            // There are barriers which are start only barriers, but some need to be legalized
            // as they are also used as end barriers
            _log.trace("There is at least one start barrier. Page start only bars {0}",
                       to_small_vector(pageStartOnlyBars));

            // All boundary task producers of common barriers need to be updated to program one of start only barriers
            // so that on those barriers all tasks from previous page finished and we are ready to reprogram them

            // Get boundary tasks producing into commonStartEndBars
            BarrierInfo::TaskSet commonStartEndBarBoundaryTaskProducers;
            for (auto commonStartEndBar : commonStartEndBars) {
                auto commonStartEndBarProducers = _barrierInfo.getBarrierProducers(commonStartEndBar);
                for (auto commonStartEndBarProducer : commonStartEndBarProducers) {
                    if (llvm::find(pageStartTasks, commonStartEndBarProducer) != pageStartTasks.end()) {
                        commonStartEndBarBoundaryTaskProducers.insert(commonStartEndBarProducer);
                    }
                }
            }

            // Pick one of start only barriers to be used to legalize those boundary tasks
            // TODO: Maybe some heuristic could be used to pick the best barrier for a task?
            auto startBarForLegalization = *std::max_element(pageStartOnlyBars.begin(), pageStartOnlyBars.end());

            for (auto commonStartEndBarBoundaryTaskProducer : commonStartEndBarBoundaryTaskProducers) {
                _barrierInfo.addProducer(startBarForLegalization, commonStartEndBarBoundaryTaskProducer);
                barrierDepsChangedHandler(startBarForLegalization, commonStartEndBarBoundaryTaskProducer);
                _log.trace("Add producer {0} to barrier {1}", commonStartEndBarBoundaryTaskProducer,
                           startBarForLegalization);
            }

            for (auto pageWithPossibleBoundaryTasksChanged : pagesWithPossibleBoundaryTasksChanged) {
                updateBoundaryTasksDataForPage(pageWithPossibleBoundaryTasksChanged);
            }

            _log.trace("Legalization completed");
            setBarProgDmaData(pageInd, pageStartOnlyBars, pageEndBars);
            _log = _log.unnest();
            continue;
        }

        // ----------------------------------------
        // Case 3: There are no start only barriers
        // ----------------------------------------
        // If there is no start only bars then pick one from the set of common start and end bars, treat it as
        // start only barrier dependencies for tasks using this barrier
        //
        //  boundaryTask0PageN-1                     boundaryTask0PageN-1
        //     |                                              |
        //  startEndBar -.......->  someTask            newStartBar -......-> someTask
        //     |                      |                                         |
        //     |                  otherEndBar   =>                         otherEndBar
        //     |                     |                                      |        |
        //  boundaryTask1PageN  boundaryTask2PageN          boundaryTask1PageN  boundaryTask2PageN
        //
        _log.trace("No start only bars");

        auto newStartBar = *std::min_element(commonStartEndBars.begin(), commonStartEndBars.end());
        _log.trace("New start bar {0}", newStartBar);

        // Remove barrier from end bars
        pageEndBars.erase(newStartBar);
        pageEndAllBars.erase(newStartBar);

        // Find all boundary tasks that is consumers of newStartBar
        auto startBarConsumersToBeLegalized = to_small_vector(_barrierInfo.getBarrierConsumers(newStartBar));
        // Leave only consumers which are boundary tasks
        startBarConsumersToBeLegalized.erase(llvm::remove_if(startBarConsumersToBeLegalized,
                                                             [&](auto taskInd) {
                                                                 return llvm::find(pageEndTasks, taskInd) ==
                                                                        pageEndTasks.end();
                                                             }),
                                             startBarConsumersToBeLegalized.end());

        // Remove boundary task consumers from start barrier
        // and reattach them to some other end barrier
        auto endBar = *std::min_element(pageEndAllBars.begin(), pageEndAllBars.end());

        _log.trace("End bar of legalization {0}", endBar);

        BarrierInfo::TaskSet startBarConsumersWhichCannotConsumeEndBar;
        for (auto startBarConsumer : startBarConsumersToBeLegalized) {
            // Check if boundary task to be legalized also updates endBar or endBar production depend on it
            // In such case it cannot be consumer of endBar and has to be legalized in a different way
            if (_barrierInfo.getUpdateBarriers(startBarConsumer).contains(endBar) ||
                isDepFromTaskToBarrier(startBarConsumer, endBar)) {
                startBarConsumersWhichCannotConsumeEndBar.insert(startBarConsumer);
                continue;
            }

            _barrierInfo.removeConsumer(newStartBar, startBarConsumer);
            barrierDepsChangedHandler(newStartBar, startBarConsumer);
            _log.trace("Remove consumer {0} from barrier {1}", startBarConsumer, newStartBar);
            _barrierInfo.addConsumer(endBar, startBarConsumer);
            barrierDepsChangedHandler(endBar, startBarConsumer);
            _log.trace("Add consumer {0} to barrier {1}", startBarConsumer, endBar);
        }
        pageEndBars.insert(endBar);

        auto endBarConsumer = _barrierInfo.getBarrierEarliestConsumer(endBar);

        // Legalize tasks which are boundary tasks and also update endBar
        // Get barriers from next page produced by such task and have endBarConsumer
        // update those barriers
        //
        //    boundaryTask -> endBar -> endBarConsumer
        //               \-------------------------------> barFromNextPage
        //
        //  Legalize to:
        //
        //    boundaryTask -> endBar -> endBarConsumer
        //                                           \---> barFromNextPage
        //
        for (auto startBarConsumer : startBarConsumersWhichCannotConsumeEndBar) {
            auto updBarsToLegalizeVec = to_small_vector(_barrierInfo.getUpdateBarriers(startBarConsumer));

            // Leave only barriers which are from next pages as only those need to be legalize
            // and updated by other boundary task - endBarConsumer
            updBarsToLegalizeVec.erase(llvm::remove_if(updBarsToLegalizeVec,
                                                       [&](auto barInd) {
                                                           return getBarrierPage(barInd) == pageInd;
                                                       }),
                                       updBarsToLegalizeVec.end());

            for (auto updBarToLegalize : updBarsToLegalizeVec) {
                _barrierInfo.addProducer(updBarToLegalize, endBarConsumer);
                barrierDepsChangedHandler(updBarToLegalize, endBarConsumer);
                _log.trace("Add producer {0} to barrier {1}", endBarConsumer, updBarToLegalize);
                _barrierInfo.removeProducer(updBarToLegalize, startBarConsumer);
                barrierDepsChangedHandler(updBarToLegalize, startBarConsumer);
                _log.trace("Remove producer {0} from barrier {1}", startBarConsumer, updBarToLegalize);
            }
            // Such task is now no longer a boundary task
            pageEndTasks.erase(startBarConsumer);
        }

        // All start boundary tasks should update newStartBar so that it is guaranteed that
        // on this barrier all tasks from previous page finished and we are ready to reprogram those barriers
        for (auto pageStartTask : pageStartTasks) {
            _log.trace("Add producer {0} to barrier {1}", pageStartTask, newStartBar);
            _barrierInfo.addProducer(newStartBar, pageStartTask);
            barrierDepsChangedHandler(newStartBar, pageStartTask);
        }

        for (auto pageWithPossibleBoundaryTasksChanged : pagesWithPossibleBoundaryTasksChanged) {
            updateBoundaryTasksDataForPage(pageWithPossibleBoundaryTasksChanged);
        }

        _log.trace("Legalization completed");
        BarrierInfo::TaskSet newPageStartBarSet;
        newPageStartBarSet.insert(newStartBar);
        setBarProgDmaData(pageInd, newPageStartBarSet, pageEndBars);

        _log = _log.unnest();
    }
}

vpux::VPURT::BarrierPagesSplitHandler::DmaProgrammingBarrierPosition
vpux::VPURT::BarrierPagesSplitHandler::getDmaProgrammingBarrierPosition(size_t pageInd) {
    VPUX_THROW_WHEN(_barProgDmaPosVec.empty(), "Barrier programming DMA positions not set");
    VPUX_THROW_UNLESS(pageInd < _pageCount - 1, "Page index {0} not within limit {1}", pageInd, _pageCount - 1);
    VPUX_THROW_WHEN(pageInd < 2 * _barrierFifoDepth - 1, "Page index {0} is expected to be programmed at bootstrap",
                    pageInd);

    return _barProgDmaPosVec[pageInd];
}

SmallVector<vpux::VPURT::BarrierPagesSplitHandler::DmaProgrammingBarrierPosition>
vpux::VPURT::BarrierPagesSplitHandler::getDmaProgrammingBarrierPositions() {
    return _barProgDmaPosVec;
}

// After inserting barrier reprogramming DMAs some subsequent tasks might need to have their page assignment
// updated so that wlmPage index never decrements on same FIFO when looking into next tasks
void vpux::VPURT::BarrierPagesSplitHandler::updateTaskPageAssignmentForQueue(size_t startTaskIndex, size_t newPageIndex,
                                                                             VPURT::TaskQueueType queueType) {
    for (size_t taskInd = startTaskIndex; taskInd < _taskPageAssignment.size(); taskInd++) {
        if (_barrierInfo.getTaskQueueType(taskInd) != queueType) {
            // Skip tasks that are not on the same queue type
            continue;
        }
        if (_taskPageAssignment[taskInd] >= newPageIndex) {
            // stop further traversal as all following task will have desired page index
            break;
        }
        VPUX_THROW_WHEN(newPageIndex > _taskPageAssignment[taskInd] + 1,
                        "New page index {0} should not be greater than current page index {1} + 1 for task {2}",
                        newPageIndex, _taskPageAssignment[taskInd], taskInd);

        _log.trace("Update task {0} page assignment from {1} to {2} for queue {3}:{4}", taskInd,
                   _taskPageAssignment[taskInd], newPageIndex, stringifyEnum(queueType.type).data(), queueType.id);
        _taskPageAssignment[taskInd] = newPageIndex;

        auto taskOp = _barrierInfo.getTaskOpAtIndex(taskInd);
        taskOp.setWlmPage(newPageIndex);
    }
}

// Check if dummy DMA is needed in case there is no DMA of this type in this page
// or if existing DMA does not meet the requirements - DMA depends on all previous page
// boundary tasks to guarantee that all barriers from previous page are fully consumed
bool VPURT::BarrierPagesSplitHandler::isDummyDmaNeeded(size_t pageInd, VPURT::TaskQueueType dmaQueueType,
                                                       std::optional<size_t> lastDmaTaskOnSameQueueInPageOpt) {
    VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(), "No boundary tasks set for page {0}",
                    pageInd);
    if (_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].find(dmaQueueType) !=
        _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].end()) {
        // There is boundary task of this type on this page. No need to insert dummy DMA
        _log.nest().trace("No need to insert dummy DMA as there is a boundary task of this type");
        return false;
    }

    // Check if in case there is a DMA of this type in a page and is not a boundary task
    // but this DMA depends on all previous page boundary tasks then there is no need to create a dummy DMA
    if (lastDmaTaskOnSameQueueInPageOpt.has_value()) {
        bool isDepFromAllBoundaryTasksToDma = true;
        auto lastDmaTaskOnSameQueueInPage = lastDmaTaskOnSameQueueInPageOpt.value();
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1].empty(),
                        "No boundary tasks set for page {0}", pageInd - 1);
        for (auto& [_, firstLastTaskInd] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1]) {
            auto lastTask = firstLastTaskInd.second;
            if (!isDepFromTaskAToTaskB(lastTask, lastDmaTaskOnSameQueueInPage)) {
                isDepFromAllBoundaryTasksToDma = false;
                break;
            }
        }
        if (isDepFromAllBoundaryTasksToDma) {
            // There is a DMA of this type in this page and it depends on all prev page boundary tasks
            // No need to insert dummy DMA
            _log.nest().trace("No need to insert dummy DMA as there is a DMA of this type in this page");
            return false;
        }
    }

    return true;
}

// Method that prepares wait barrier data for dummy DMA
// In case of crossing control graph block boundary it can also legalize prev page boundary tasks update barriers
// so that wait barriers of dummy DMA will depend on those tasks
vpux::BarrierInfo::TaskSet VPURT::BarrierPagesSplitHandler::getDummyDmaWaitBars(size_t pageInd) {
    // As wait barrier use wait barrier of earliest boundary task on this page
    // WLM page split guarantees that this wait barrier is updated by all tasks from previous page
    auto boundaryTasks = getFirstBoundaryTasksForPage(pageInd);
    VPUX_THROW_WHEN(boundaryTasks.empty(), "No boundary tasks set for page {0}", pageInd);
    auto boundaryTask = boundaryTasks.front();

    auto dummyDmaProposedWaitBars = _barrierInfo.getWaitBarriers(boundaryTask);

    if (dummyDmaProposedWaitBars.empty()) {
        auto taskWithWaitBarOnSamePageOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(boundaryTask);
        VPUX_THROW_UNLESS(taskWithWaitBarOnSamePageOpt.has_value(), "No wait barriers for task {0}(page {1})",
                          boundaryTask, _taskPageAssignment[boundaryTask]);

        boundaryTask = taskWithWaitBarOnSamePageOpt.value();
        dummyDmaProposedWaitBars = _barrierInfo.getWaitBarriers(boundaryTask);
    }

    // Check if wait barrier is used by control graph sync point then it cannot be used as wait barrier
    // of dummy DMA because there will be no update barrier to use that will not break control graph split
    bool isWaitBarUsedBySyncPoint = false;
    for (auto waitBar : dummyDmaProposedWaitBars) {
        for (auto waitBarUser : _barrierInfo.getBarrierConsumers(waitBar)) {
            isWaitBarUsedBySyncPoint = _barrierInfo.isSyncPoint(waitBarUser);
            if (isWaitBarUsedBySyncPoint) {
                _log.nest().trace("Wait barrier {0} is used by control graph sync point {1}", waitBar, waitBarUser);
                break;
            }
        }
        if (isWaitBarUsedBySyncPoint) {
            break;
        }
    }

    // In the case of wait barrier used by sync point use different wait barriers for dummy DMA
    if (isWaitBarUsedBySyncPoint) {
        // Store information about prev page boundary tasks that will need to be legalized - they
        // no longer can update barrier consumed by sync task but update barrier that will
        // be consumed by dummy DMA
        SmallVector<size_t> prevPageBoundaryTasksToHaveUpdateBarLegalized;

        auto prevPageInd = pageInd - 1;
        _log.nest().trace("Change wait barriers to be update barrier of page {0} boundary tasks", prevPageInd);
        dummyDmaProposedWaitBars.clear();

        // As wait barriers use update barriers of boundary tasks from previous page. After that dummy DMA
        // may have more than 1 wait barrier but since all DMA tasks are expected to be enqueued at bootstrap
        // this is not a problem for enqueue algorithm
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[prevPageInd].empty(),
                        "No boundary tasks set for page {0}", prevPageInd);
        for (auto& [_, firstLastTaskIndPair] : _firstAndLastBoundaryTaskForEachPagePerFifo[prevPageInd]) {
            auto prevPageBoundaryTask = firstLastTaskIndPair.second;
            auto prevPageBoundaryTaskUpdBars = _barrierInfo.getUpdateBarriers(prevPageBoundaryTask);
            _log.nest(2).trace("Page {0} boundary task {1} update barriers {2}", prevPageInd, prevPageBoundaryTask,
                               to_small_vector(prevPageBoundaryTaskUpdBars));

            // Check if barrier is consumed by sync point task. In that case it cannot be used
            // as later this barrier will be used as update barrier of dummy DMA
            auto prevPageBoundaryTaskValidUpdBars = prevPageBoundaryTaskUpdBars;
            for (auto prevPageBoundaryTaskUpdBar : prevPageBoundaryTaskUpdBars) {
                if (llvm::any_of(_barrierInfo.getBarrierConsumers(prevPageBoundaryTaskUpdBar), [&](auto userTask) {
                        return _barrierInfo.isSyncPoint(userTask);
                    })) {
                    prevPageBoundaryTaskValidUpdBars.erase(prevPageBoundaryTaskUpdBar);
                }
            }

            if (prevPageBoundaryTaskValidUpdBars.empty()) {
                _log.nest(2).trace("No update barrier not used by sync task for task {0}", prevPageBoundaryTask);
                // If there is no update barrier from prev page boundary task that can be used
                // then need to legalize prev page boundary task to also update some other barrier
                // that later dummy DMA will depend on
                prevPageBoundaryTasksToHaveUpdateBarLegalized.push_back(prevPageBoundaryTask);
                continue;
            }

            // Just 1 update barrier for each boundary task is enough. Pick one with smallest index
            // so that dummy DMA will depend on earliest barriers possible
            dummyDmaProposedWaitBars.insert(*std::min_element(prevPageBoundaryTaskValidUpdBars.begin(),
                                                              prevPageBoundaryTaskValidUpdBars.end()));
        }

        VPUX_THROW_WHEN(dummyDmaProposedWaitBars.empty(), "No wait barriers for dummy DMA in page {0} found", pageInd);

        // In case of legalizing prevPageBoundaryTasksToHaveUpdateBarLegalized tasks,
        // from all identified wait barriers to be used for dummy DMA pick the latest one which will
        // be used as an update barrier for pref page boundary task to guarantee that dummy DMA execution
        // depends on all prev page boundary tasks.
        // Use barrier with highest index to prevent from picking barrier which is consumed earlier than
        // given boundary task is located what could create cyclic dependency and compilation error
        auto latestTaskWaitBar = *std::max_element(dummyDmaProposedWaitBars.begin(), dummyDmaProposedWaitBars.end());
        for (auto boundaryTaskToHaveUpdateBarLegalized : prevPageBoundaryTasksToHaveUpdateBarLegalized) {
            _log.nest(2).trace("Legalize update barrier for task {0}", boundaryTaskToHaveUpdateBarLegalized);
            // Locate update barrier consumed by sync point. Prev page boundary task that updates this barrier
            // now will need to update latestTaskWaitBar to guarantee that there is a dependency
            // from prev page boundary task to dummy DMA
            auto updBarConsumedBySyncTask = *llvm::find_if(
                    _barrierInfo.getUpdateBarriers(boundaryTaskToHaveUpdateBarLegalized), [&](auto barInd) {
                        return llvm::any_of(_barrierInfo.getBarrierConsumers(barInd), [&](auto userTask) {
                            return _barrierInfo.isSyncPoint(userTask);
                        });
                    });
            _log.nest(3).trace("Change update barrier from {0} to {1}", updBarConsumedBySyncTask, latestTaskWaitBar);
            _barrierInfo.removeProducer(updBarConsumedBySyncTask, boundaryTaskToHaveUpdateBarLegalized);
            _barrierInfo.addProducer(latestTaskWaitBar, boundaryTaskToHaveUpdateBarLegalized);
        }
    }

    return dummyDmaProposedWaitBars;
}

// Based on wait barriers of dummy DMA and last DMA task on the same queue in this page (if exists)
// return after which task dummy DMA can be inserted
size_t VPURT::BarrierPagesSplitHandler::getDummyDmaInsertionPoint(
        vpux::BarrierInfo::TaskSet& dummyDmaProposedWaitBars, std::optional<size_t> lastDmaTaskOnSameQueueInPageOpt) {
    // For finding insertion point of dummy DMA take into account:
    // 1. lastWaitBarrierProducer - last task (max index) that updates barriers from dummyDmaProposedWaitBars
    // 2. lastDmaTaskOnSameQueueInPage - last DMA index on the same queue in this page
    // Dummy DMA needs to be inserted after both above
    // insertAfter = max(lastWaitBarrierProducer, lastDmaTaskOnSameQueueInPage)
    auto lastWaitBarrierProducer = _barrierInfo.getBarriersLatestProducer(dummyDmaProposedWaitBars);

    auto insertionPoint = lastWaitBarrierProducer;
    if (lastDmaTaskOnSameQueueInPageOpt.has_value()) {
        insertionPoint = std::max(insertionPoint, lastDmaTaskOnSameQueueInPageOpt.value());
    }

    return insertionPoint;
}

// Method that prepares update barrier data for dummy DMA
vpux::BarrierInfo::TaskSet VPURT::BarrierPagesSplitHandler::getDummyDmaUpdateBars(
        size_t pageInd, size_t insertionPoint, SmallVector<std::pair<size_t, size_t>>& firstAndLastBarIndPerPage) {
    // As update barrier use some barrier from next page. Pick one which has
    // earliest consumer later than insertion point of dummy DMA
    // Scan range of barriers from the end till beginning of next page. In case of
    // last page start from last barrier except the final barrier
    // TODO: Usage of update barrier might no longer be needed after E#167504 is implemented
    auto barIndStart = firstAndLastBarIndPerPage[pageInd + 1].first;
    auto barIndEnd = firstAndLastBarIndPerPage[pageInd + 1].second;
    if (pageInd + 1 == _pageCount - 1) {
        // If this is last page do not check final barrier
        barIndEnd--;
    }
    _log.nest().trace("Update barriers search in range {0} - {1}", barIndStart, barIndEnd);
    BarrierInfo::TaskSet dummyDmaProposedUpdateBars;
    auto dummyDmaBlockIndex = _barrierInfo.getControlGraphBlockIndex(insertionPoint + 1);
    for (int barInd = barIndEnd; barInd >= static_cast<int>(barIndStart); barInd--) {
        auto earliestConsumer = _barrierInfo.getBarrierEarliestConsumer(barInd);
        if (earliestConsumer > insertionPoint) {
            auto earliestConsumerBlockIndex = _barrierInfo.getControlGraphBlockIndex(earliestConsumer);
            if (earliestConsumerBlockIndex > dummyDmaBlockIndex) {
                // If identified consumer is on different control block, then
                // use sync task wait barrier as update barrier to not break
                // control graph split requirement
                auto syncTaskOpt = _barrierInfo.getControlGraphSyncPoint(insertionPoint);
                VPUX_THROW_UNLESS(syncTaskOpt.has_value(), "No control block sync task for task {0}", insertionPoint);

                auto syncTaskWaitBars = _barrierInfo.getWaitBarriers(syncTaskOpt.value());
                VPUX_THROW_UNLESS(!syncTaskWaitBars.empty(), "No wait barriers for control graph sync task {0} found",
                                  syncTaskOpt.value());
                auto newBarInd = *std::max_element(syncTaskWaitBars.begin(), syncTaskWaitBars.end());
                _log.nest(2).trace("Change update barrier {0} to {1} due to crossing control graph block "
                                   "boundary set by task {2}",
                                   barInd, newBarInd, syncTaskOpt.value());
                dummyDmaProposedUpdateBars.insert(newBarInd);
            } else {
                dummyDmaProposedUpdateBars.insert(barInd);
            }
            break;
        }
    }
    VPUX_THROW_WHEN(dummyDmaProposedUpdateBars.empty(), "No update barriers for dummy DMA in page {0} found", pageInd);

    return dummyDmaProposedUpdateBars;
}

// Get information for each page about last DMA of each type
SmallVector<mlir::DenseMap<VPURT::TaskQueueType, size_t>> VPURT::BarrierPagesSplitHandler::getLastDmaOfTypePerPage() {
    SmallVector<mlir::DenseMap<VPURT::TaskQueueType, size_t>> lastDmaOfTypePerPage(_pageCount);

    for (size_t taskInd = 0; taskInd < _taskPageAssignment.size(); taskInd++) {
        auto pageInd = _taskPageAssignment[taskInd];
        auto taskQueueType = _barrierInfo.getTaskQueueType(taskInd);
        if (taskQueueType.type == VPU::ExecutorKind::DMA_NN) {
            lastDmaOfTypePerPage[pageInd][taskQueueType] = taskInd;
        }
    }
    return lastDmaOfTypePerPage;
}

// To enable enqueue of all DMA tasks at bootstrap as a single link-list each page needs to have a DMA task.
// If such is not present dummy DMA needs to be inserted. This function returns information on where such
// dummy DMAs need to be inserted.
// Requirement is following:
// If there is no DMAx in PageN whose execution guarantees that barriers from PageN-1 have been consumed before
// and that there are DMAs of this type in some later pages, then we need to insert task of DMAx in PageN
// and this DMAx needs to depend on all boundary tasks from PageN-1/PageN border.
SmallVector<VPURT::BarrierPagesSplitHandler::DummyDmaInsertionData>
VPURT::BarrierPagesSplitHandler::getAndLegalizeDummyDmaInsertionData() {
    _log.trace("Getting dummy DMA insertion data");
    SmallVector<DummyDmaInsertionData> dummyDmaInsertionDataVec;

    SmallVector<std::pair<VPURT::TaskQueueType, size_t>> dmaQueueTypesAndLastPageInd;

    // Get all DMA queue types and last page index for DMA on this queue
    // There is no need in inserting dummy DMAs if there are no more DMAs
    // of this type in subsequent pages
    for (auto queueType : _barrierInfo.getNonEmptyTaskQueueTypes()) {
        if (queueType.type == VPU::ExecutorKind::DMA_NN) {
            _barrierInfo.getLastTaskForQueueType(queueType);
            size_t lastPage = _taskPageAssignment[_barrierInfo.getLastTaskForQueueType(queueType)];

            // In case of DMA Port:0 Channel:DDR which is also used for WLM enqueue DMAs
            // it needs to be guaranteed that there is such DMA in each page so that
            // enqueue DMAs can be inserted afterwards and link listed to rest of DMAs
            if (queueType.id == getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)) {
                lastPage = _taskPageAssignment[_barrierInfo.getNumOfTasks() - 1];
            }

            dmaQueueTypesAndLastPageInd.push_back(std::make_pair(queueType, lastPage));
        }
    }

    // Get information for each page about last DMA of each type
    // This information will be later used to check if dummy DMA is really needed
    // and to find insertion point of dummy DMAs
    auto lastDmaOfTypePerPage = getLastDmaOfTypePerPage();

    // Get information for each page about first and last barrier index
    // This is going to be used to find update barrier for dummy DMA
    // TODO: This should no longer be needed after E#167504 is implemented
    SmallVector<std::pair<size_t, size_t>> firstAndLastBarIndPerPage;
    firstAndLastBarIndPerPage.resize(_pageCount);
    firstAndLastBarIndPerPage[0].first = 0;
    for (size_t barInd = 1; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto pageInd = getBarrierPage(barInd);
        if (firstAndLastBarIndPerPage[pageInd].first == 0) {
            firstAndLastBarIndPerPage[pageInd].first = barInd;
        }
        firstAndLastBarIndPerPage[pageInd].second = barInd;
    }

    // Iterate each page and check if there is a DMA boundary task of each type
    // Skip first page as the earliest place where we might need to insert dummy DMA is Page1 to prevent
    // tasks from Page2 to potentially execute prematurely in Page0
    // Skip last page as there are is no need to insert dummy DMA there as there are no pages afterwards
    for (size_t pageInd = 1; pageInd < _pageCount - 1; pageInd++) {
        // Check if given page has boundary task of each DMA type
        for (auto [dmaQueueType, lastPageInd] : dmaQueueTypesAndLastPageInd) {
            if (pageInd >= lastPageInd) {
                continue;
            }

            std::optional<size_t> lastDmaTaskOnSameQueueInPageOpt =
                    (lastDmaOfTypePerPage[pageInd].find(dmaQueueType) != lastDmaOfTypePerPage[pageInd].end())
                            ? std::make_optional<size_t>(lastDmaOfTypePerPage[pageInd][dmaQueueType])
                            : std::nullopt;

            _log.trace("Check page {0} for DMA of type {1}:{2}", pageInd, stringifyEnum(dmaQueueType.type).data(),
                       dmaQueueType.id);

            if (!isDummyDmaNeeded(pageInd, dmaQueueType, lastDmaTaskOnSameQueueInPageOpt)) {
                continue;
            }

            _log = _log.nest();
            _log.trace("Prepare data for dummy DMA");

            DummyDmaInsertionData dummyDmaInsertionData;
            dummyDmaInsertionData.pageInd = pageInd;
            dummyDmaInsertionData.queueType = dmaQueueType;

            auto dummyDmaProposedWaitBars = getDummyDmaWaitBars(pageInd);
            dummyDmaInsertionData.waitBars = to_small_vector(dummyDmaProposedWaitBars);

            auto insertionPoint = getDummyDmaInsertionPoint(dummyDmaProposedWaitBars, lastDmaTaskOnSameQueueInPageOpt);
            dummyDmaInsertionData.insertAfter = insertionPoint;

            _log.nest().trace("Insert after op {0}", dummyDmaInsertionData.insertAfter);
            _log.nest().trace("Wait barriers: {0}", dummyDmaInsertionData.waitBars);

            auto dummyDmaProposedUpdateBars = getDummyDmaUpdateBars(pageInd, insertionPoint, firstAndLastBarIndPerPage);
            dummyDmaInsertionData.updateBars = to_small_vector(dummyDmaProposedUpdateBars);

            _log.nest().trace("Update barriers: {0}", dummyDmaInsertionData.updateBars);

            // Below exception was added to catch case when same barrier is used as both wait and update barrier
            // what creates cyclic dependency. This is caused by adjusting update barrier to not break control graph
            // split. Revisit this once update barriers are no longer needed when E#162444 and E#166675 is implemented
            VPUX_THROW_UNLESS(llvm::set_intersection(dummyDmaProposedWaitBars, dummyDmaProposedUpdateBars).empty(),
                              "Common wait ({0}) and update barrier ({1})", dummyDmaInsertionData.waitBars,
                              dummyDmaInsertionData.updateBars);

            dummyDmaInsertionDataVec.push_back(dummyDmaInsertionData);
            _log = _log.unnest();
        }
    }

    return dummyDmaInsertionDataVec;
}

// Prepare data for inserting dummy barriers. They are needed to be placed
// in pages which use less than half of available physical barriers. To make barrier
// programming DMAs simple and able to always refill 4 entries for all physical barriers
// as a single transaction, each page except last two need to always use exactly half of
// physical barriers. Dummy barriers will be placed in parallel to existing barriers
// what will not have any impact on performance of schedule.
SmallVector<VPURT::BarrierPagesSplitHandler::DummyBarrierData>
VPURT::BarrierPagesSplitHandler::getDummyBarriersInsertionData() {
    _log.trace("Getting dummy barriers data");
    SmallVector<DummyBarrierData> dummyBarrierDataVec;

    if (_pageCount <= 2) {
        _log.trace("No need to insert dummy barriers if model has {0} <= 2 pages", _pageCount);
        return dummyBarrierDataVec;
    }

    SmallVector<size_t> numberOfBarriersPerPage(_pageCount, 0);
    SmallVector<size_t> lastBarrierIndexPerPage(_pageCount, 0);
    // Iterate each barrier and count number of barriers per page and store information about
    // last barrier index
    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto pageInd = getBarrierPage(barInd);
        numberOfBarriersPerPage[pageInd]++;
        lastBarrierIndexPerPage[pageInd] = barInd;
    }

    // Iterate each page and check the number of barriers
    // Skip last two pages, which are last pages for two halves of physical barrier set.
    // There is no need to insert dummy barriers there as those entries can be filled by dummy values
    // when programming barrier FIFOs using barrier programming DMA
    _log = _log.nest();
    for (size_t pageInd = 0; pageInd < _pageCount - 2; pageInd++) {
        _log.trace("Page {0} barrier count: {1}", pageInd, numberOfBarriersPerPage[pageInd]);
        for (size_t barrierIndexToAdd = numberOfBarriersPerPage[pageInd]; barrierIndexToAdd < _pageSize;
             barrierIndexToAdd++) {
            _log.nest().trace("Missing barrier. Prepare new one {0}", barrierIndexToAdd);

            DummyBarrierData dummyBarrierData;
            dummyBarrierData.pageInd = pageInd;
            dummyBarrierData.insertAfter = lastBarrierIndexPerPage[pageInd];
            dummyBarrierData.consumer = *(_barrierInfo.getBarrierConsumers(lastBarrierIndexPerPage[pageInd]).begin());
            dummyBarrierData.producer = *(_barrierInfo.getBarrierProducers(lastBarrierIndexPerPage[pageInd]).begin());
            _log.nest().trace("New barrier data: insert after bar {0}, producer {1}, consumer {2}",
                              dummyBarrierData.insertAfter, dummyBarrierData.producer, dummyBarrierData.consumer);

            dummyBarrierDataVec.push_back(dummyBarrierData);
        }
    }
    _log = _log.unnest();

    return dummyBarrierDataVec;
}

void vpux::VPURT::BarrierPagesSplitHandler::initPrevPhysBarrierData(SmallVector<size_t>& barrierToPidVec) {
    size_t numOfBarriers = _barrierInfo.getNumOfBarrierOps();

    VPUX_THROW_UNLESS(numOfBarriers == barrierToPidVec.size(), "Not matching number of barriers {0} != {1}",
                      numOfBarriers, barrierToPidVec.size());

    _barrierPidPrevUsageVec.resize(numOfBarriers);
    SmallVector<std::optional<size_t>> lastBarUsingPid(_pageSize * 2);

    for (size_t vid = 0; vid < numOfBarriers; vid++) {
        auto pid = barrierToPidVec[vid];
        _barrierPidPrevUsageVec[vid] = lastBarUsingPid[pid];
        lastBarUsingPid[pid] = vid;
    }
}

// For each barrier find index of previous barrier using same PID
void vpux::VPURT::BarrierPagesSplitHandler::initPrevPhysBarrierData(mlir::func::FuncOp func) {
    // Get information for each barrier about previous barrier that used same PID
    size_t numOfBarriers = _barrierInfo.getNumOfBarrierOps();

    SmallVector<size_t> barrierToPidVec(numOfBarriers);

    func.walk([&](VPURT::ConfigureBarrierOp barOp) {
        size_t pid = barOp.getId();
        auto vid = _barrierInfo.getIndex(barOp);
        barrierToPidVec[vid] = pid;
    });

    initPrevPhysBarrierData(barrierToPidVec);
}

// This method  enqueue information in case fo Full WLM for each task
// Return enqueue barrier for each task
// TODO: In next stage this method will return information about enqueue DMAs (E#170833)
SmallVector<std::optional<size_t>> vpux::VPURT::BarrierPagesSplitHandler::prepareEnqueueDmaBarForFullWlm(
        const mlir::DenseSet<vpux::VPU::ExecutorKind>& executorEnqAtBootstrap) {
    VPUX_THROW_WHEN(_barrierPidPrevUsageVec.empty(), "Barrier PID previous usage vector is not initialized.");
    SmallVector<std::optional<size_t>> tasksEnqBar(_barrierInfo.getNumOfTasks());

    const VPURT::TaskQueueType dmaP0ChDdrQueueType = {VPU::ExecutorKind::DMA_NN,
                                                      getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};
    auto lastDmaOfTypePerPage = getLastDmaOfTypePerPage();

    for (auto& [queueType, taskVec] : _taskQueueTypeMap) {
        _log.trace("Enqueue tasks for {0}:{1}", VPU::stringifyExecutorKind(queueType.type), queueType.id);

        // Check if for given queue it is requested to enqueue all tasks at bootstrap
        // In that case skip processing of this queue as default _tasksEnqBar value is nullopt = BOOTSTRAP
        if (executorEnqAtBootstrap.contains(queueType.type)) {
            _log.nest().trace("Enqueue task from that queue at bootstrap");
            continue;
        }

        // DMA tasks can be enqueued at bootstrap as they do not require
        // descriptor fetching but DPU and SHV need to be enqueued at start barrier as earliest point
        // TODO: This theoretically is no longer needed for Full WLM if we make sure descriptor fetching DMA is
        // before enqueue DMA. To be removed in future
        bool supportEnqAtBootstrap = (queueType.type == VPU::ExecutorKind::DMA_NN);

        // Earliest enqueue barrier for queue
        std::optional<size_t> earliestEnqBarOpt = std::nullopt;
        if (!supportEnqAtBootstrap) {
            VPUX_THROW_UNLESS(_startBarrierIndex.has_value(), "Start barrier index is not set");
            earliestEnqBarOpt = _startBarrierIndex;
        }

        // Check if there is a need to check enqueue of each DPU if it would not block
        // submission of DPUs from SHV
        bool dpuEnqCheckForShv = queueType.type == VPU::ExecutorKind::DPU && !_shvTasksWithDpuPerTile.empty() &&
                                 !_shvTasksWithDpuPerTile[queueType.id].empty();
        if (dpuEnqCheckForShv) {
            _log.trace("There are {0} SHV tasks which submit DPU. DPU[{1}] enqueue needs to take that into account",
                       _shvTasksWithDpuPerTile[queueType.id].size(), queueType.id);
        }

        std::optional<size_t> previousEnqBarOpt = std::nullopt;
        std::optional<size_t> previousTaskIndOpt = std::nullopt;
        std::optional<size_t> prevTaskPage = std::nullopt;

        _log = _log.nest();
        // Iterate over all tasks of this queue type and find enqueue barrier for each task
        for (auto taskInd : taskVec) {
            auto taskPage = _taskPageAssignment[taskInd];

            auto waitBars = _barrierInfo.getWaitBarriers(taskInd);

            std::optional<size_t> enqBarOpt = std::nullopt;

            _log.trace("Find enqueue for task {0}(page {1}) with wait bars {2}", taskInd, taskPage,
                       to_small_vector(waitBars));

            // Find enqueue barrier based on conditions
            if (taskPage < 2) {
                // Case 1:
                // Tasks from Page 0 and 1 can be enqueued at bootstrap
                enqBarOpt = earliestEnqBarOpt;
                _log.nest().trace("Page 0 and 1 tasks can be enqueued at {0}",
                                  (enqBarOpt.has_value() ? std::to_string(enqBarOpt.value()) : "BOOTSTRAP"));
            } else if (prevTaskPage.has_value() && prevTaskPage.value() == taskPage) {
                // Case 2:
                // If task is on same page as previous task then enqueue it together
                enqBarOpt = previousEnqBarOpt;
                _log.nest().trace("Task has same page as previous. Enqueue together with previous: {0}",
                                  (enqBarOpt.has_value() ? std::to_string(enqBarOpt.value()) : "BOOTSTRAP"));
            } else {
                // Case 3:
                // Find enqueue barrier based on task wait barriers
                SmallVector<size_t> prevBars;
                for (auto barInd : waitBars) {
                    auto prevBar = _barrierPidPrevUsageVec[barInd];
                    if (prevBar.has_value()) {
                        prevBars.push_back(prevBar.value());
                    }
                }

                if (prevBars.empty()) {
                    // Case 3a:
                    // If task has no previous instances of wait barriers
                    if (previousEnqBarOpt.has_value()) {
                        _log.nest().trace("Task has no previous barrier instances. Enqueue together with previous");
                        enqBarOpt = previousEnqBarOpt;
                    } else {
                        _log.nest().trace("Task has no previous barrier instances. Enqueue at bootstrap/startbarrier");
                        enqBarOpt = earliestEnqBarOpt;
                    }
                }

                if (!enqBarOpt.has_value() && previousEnqBarOpt.has_value() && previousTaskIndOpt.has_value()) {
                    // Case 3b:
                    // Check if task can be enqueued with previous by checking if there is dependency from all previous
                    // instances of wait barriers users to previous task
                    bool canBeMerged = true;
                    auto prevTask = previousTaskIndOpt.value();
                    _log.nest().trace("Check if task can be merged with previous - {0}(page {1})", prevTask,
                                      _taskPageAssignment[prevTask]);
                    for (auto prevBar : prevBars) {
                        auto prevBarUsers = _barrierInfo.getBarrierConsumers(prevBar);
                        _log.nest(2).trace("Check dependencies from prev bar {0} users: {1} to prevTask {2}", prevBar,
                                           to_small_vector(prevBarUsers), prevTask);
                        for (auto prevBarUser : prevBarUsers) {
                            if (!isDepFromTaskAToTaskB(prevBarUser, prevTask)) {
                                _log.nest(2).trace("Cannot be enqueued with previous because there is no dependency "
                                                   "from {0} to {1}",
                                                   prevBarUser, prevTask);
                                canBeMerged = false;
                                break;
                            }
                        }
                        if (!canBeMerged) {
                            break;
                        }
                    }

                    if (canBeMerged) {
                        enqBarOpt = previousEnqBarOpt;
                        _log.nest().trace("Can be enqueued with previous task on barrier {0}",
                                          (enqBarOpt.has_value() ? std::to_string(enqBarOpt.value()) : "BOOTSTRAP"));
                    }
                }

                if (!enqBarOpt.has_value()) {
                    // Case 3c:
                    // Safe enqueue for task in PageN is to use last DMA of Page N-1 as previous pass which insert dummy
                    // DMAs makes sure such DMA will execute after all barriers from PageN-2 has been consumed
                    // TODO: Experiment with more optimal solutions and find earlier possible enqueue point.
                    // This will be needed in case this method would inject enqueue DMAs directly (E#170833)
                    auto lastDmaP0ChDdr = lastDmaOfTypePerPage[taskPage - 1][dmaP0ChDdrQueueType];
                    _log.nest().trace("Last DMA P0 CH:DDR on prev page - {0}(page {1})", lastDmaP0ChDdr,
                                      _taskPageAssignment[lastDmaP0ChDdr]);

                    auto lastDmaP0ChDdrWaitBars = _barrierInfo.getWaitBarriers(lastDmaP0ChDdr);

                    if (lastDmaP0ChDdrWaitBars.empty()) {
                        auto prevTaskOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(lastDmaP0ChDdr);
                        if (prevTaskOpt.has_value()) {
                            lastDmaP0ChDdrWaitBars = _barrierInfo.getWaitBarriers(prevTaskOpt.value());
                            _log.nest(2).trace("Use wait barriers of task {0}(page {1}) - {2}", prevTaskOpt.value(),
                                               _taskPageAssignment[prevTaskOpt.value()],
                                               to_small_vector(lastDmaP0ChDdrWaitBars));
                        }
                    }

                    VPUX_THROW_UNLESS(!lastDmaP0ChDdrWaitBars.empty(),
                                      "No wait barriers for last DMA {0}(page {1}) found", lastDmaP0ChDdr,
                                      taskPage - 1);

                    enqBarOpt = *std::min_element(lastDmaP0ChDdrWaitBars.begin(), lastDmaP0ChDdrWaitBars.end());

                    auto enqBarPage = getBarrierPage(enqBarOpt.value());
                    VPUX_THROW_WHEN(enqBarPage >= taskPage,
                                    "Enqueue barrier {0}{page {1}} is not on earlier page than task", enqBarOpt.value(),
                                    enqBarPage);

                    _log.nest().trace("Enqueue task at {0}(page {1})", enqBarOpt.value(), enqBarPage);
                }
            }

            // Check if enqueue barrier needs to be delayed because of SHV task submitting DPU
            if (dpuEnqCheckForShv) {
                for (auto shvTaskInd : _shvTasksWithDpuPerTile[queueType.id]) {
                    if (isDepFromTaskAToTaskB(shvTaskInd, taskInd)) {
                        // DPU task depends on SHV. Check if DPU enq barrier is after SHV
                        _log.nest().trace("DPU task {0} with enqueue at {1} depends on SHV task {2} which submits DPU",
                                          taskInd, enqBarOpt.value(), shvTaskInd);
                        if (!isDepFromTaskToBarrier(shvTaskInd, enqBarOpt.value())) {
                            // If enqueue is not after SHV task completion then delay it
                            // Check on which barrier to delay. Currently it is guaranteed by previous
                            // passes that SHV with DPU will have following sequence:
                            //  .. -> SHV(DPU) -> BAR -> SyncDMA -> ...
                            // In such case it is safe to delay on BAR barrier
                            enqBarOpt = *_barrierInfo.getUpdateBarriers(shvTaskInd).begin();
                            _log.nest().trace(
                                    "Delay enqueue of task {0} to {1} due to dependency on SHV task {2} which "
                                    "submits DPU",
                                    taskInd, enqBarOpt.value(), shvTaskInd);
                        }
                    }
                }
            }

            tasksEnqBar[taskInd] = enqBarOpt;
            previousEnqBarOpt = enqBarOpt;
            previousTaskIndOpt = taskInd;
            prevTaskPage = taskPage;
        }
        _log = _log.unnest();
    }
    return tasksEnqBar;
}

void vpux::VPURT::BarrierPagesSplitHandler::cleanupRedundantBarriers() {
    _log.trace("Cleaning up redundant barriers");
    // Cleanup redundant barriers
    // If barrier has no consumers, remove its producers. Ignore final barrier (last barrier)
    // If barrier has no producers, remove its consumers
    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps() - 1; barInd++) {
        if (_barrierInfo.getBarrierConsumers(barInd).empty()) {
            _log.trace("No consumers for barrier {0}(page {1})", barInd, getBarrierPage(barInd));
            auto barProdTasks = _barrierInfo.getBarrierProducers(barInd);
            for (auto barProdTask : barProdTasks) {
                _log.trace("Remove producer {0}(page {1}) from barrier {2}(page {3})", barProdTask,
                           _taskPageAssignment[barProdTask], barInd, getBarrierPage(barInd));
                _barrierInfo.removeProducer(barInd, barProdTask);
            }
        }
        if (_barrierInfo.getBarrierProducers(barInd).empty()) {
            _log.trace("No producer for barrier {0}(page {1})", barInd, getBarrierPage(barInd));
            auto barConsTasks = _barrierInfo.getBarrierConsumers(barInd);
            for (auto barConsTask : barConsTasks) {
                _log.trace("Remove consumer {0}(page {1}) from barrier {2}(page {3})", barConsTask,
                           _taskPageAssignment[barConsTask], barInd, getBarrierPage(barInd));
                _barrierInfo.removeConsumer(barInd, barConsTask);
            }
        }
    }
}

// Verify all tasks access barriers only from their page and next page
void vpux::VPURT::BarrierPagesSplitHandler::verifyTaskBarrierPagesAreValid() {
    _log.trace("Verifying task barrier pages are valid");
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto waitBars = _barrierInfo.getWaitBarriers(taskInd);
        auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);

        llvm::DenseSet<size_t> waitPages;
        llvm::for_each(waitBars, [&](auto& bar) {
            waitPages.insert(getBarrierPage(bar));
        });

        llvm::DenseSet<size_t> updatePages;
        llvm::for_each(updateBars, [&](auto& bar) {
            updatePages.insert(getBarrierPage(bar));
        });

        auto taskPage = _taskPageAssignment[taskInd];

        if (!waitPages.empty()) {
            auto maxWaitPage = *std::max_element(waitPages.begin(), waitPages.end());
            VPUX_THROW_UNLESS(maxWaitPage <= taskPage, "Task {0} page {1} has wait barrier from page {2}", taskInd,
                              taskPage, maxWaitPage);
        }

        if (!updatePages.empty()) {
            auto minUpdatePage = *std::min_element(updatePages.begin(), updatePages.end());
            VPUX_THROW_UNLESS(minUpdatePage >= taskPage, "Task {0} page {1} has update barrier from page {2}", taskInd,
                              taskPage, minUpdatePage);
        }
    }
}

// Verification function that checks if after legalization and modification of schedule
// no cyclic dependency was created. This is just to catch such issue earlier, otherwise this problem
// would show up later during task and barrier reordering or barrier simulation
void vpux::VPURT::BarrierPagesSplitHandler::verifyNoCyclicDeps() {
    _log.trace("Verifying no cyclic deps");
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);

        for (auto bar : updateBars) {
            auto barConsumers = _barrierInfo.getBarrierConsumers(bar);
            for (auto barConsumer : barConsumers) {
                if (barConsumer > taskInd) {
                    continue;
                }

                if (taskInd == barConsumer) {
                    VPUX_THROW("Cyclic dependency detected - task {0}(page {1}) updates barrier {2}(page {3}) which is "
                               "consumed by the same "
                               "task",
                               taskInd, _taskPageAssignment[taskInd], bar, getBarrierPage(bar));
                }

                if (isDepFromTaskAToTaskB(barConsumer, taskInd)) {
                    VPUX_THROW("Cyclic dependency detected - task {0}(page {1}) updates barrier {2}(page {3}) which is "
                               "consumed by task "
                               "{4}(page {5}) which task {0}(page {1}) depends on",
                               taskInd, _taskPageAssignment[taskInd], bar, getBarrierPage(bar), barConsumer,
                               _taskPageAssignment[barConsumer]);
                }
            }
        }
    }
}

// Perform complete legalization of schedule for barrier pages split
// - legalize long dependencies of tasks
// - ensure boundary tasks from neighbor pages are dependent
void vpux::VPURT::BarrierPagesSplitHandler::legalizeScheduleForBarrierPagesSplit() {
    _log.trace("Legalizing schedule for barrier pages split");
    if (!areNoDepsGoingBeyondNeighborPage()) {
        auto tasksToLegalize = getTasksWithNonAdjacentPageDependencyToLegalize();
        legalizeNonAdjacentPageDependencies(tasksToLegalize);
    }

    if (!areBoundaryTasksFromNeighborPagesDependent()) {
        auto boundaryTaskPairsMissingDepInBetween = getBoundaryTaskPairsMissingDepInBetween();
        legalizeDepsForBoundaryTasks(boundaryTaskPairsMissingDepInBetween);
    }

    cleanupRedundantBarriers();
    verifyTaskBarrierPagesAreValid();
    verifyNoCyclicDeps();
    _log.trace("Successfully legalized schedule for barrier pages split");
}

void vpux::VPURT::BarrierPagesSplitHandler::updateIR() {
    _barrierInfo.updateIR();
}

// Helper method for unit testing BarrierPagesSplitHandler
BarrierInfoMaps vpux::VPURT::BarrierPagesSplitHandler::getBarrierMaps() {
    return vpux::getBarrierMaps(_barrierInfo);
}
