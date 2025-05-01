//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/barrier_pages_split.hpp"
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
    _pageCount = (_barrierInfo.getNumOfBarrierOps() + _pageSize - 1) / _pageSize;
}

// Below constructor is meant to be used only for unit testing purpose
vpux::VPURT::BarrierPagesSplitHandler::BarrierPagesSplitHandler(
        BarrierInfoTest& barrierInfoTest, std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueTypeMap,
        size_t pageSize, size_t _barrierFifoDepth, Logger log)
        : _barrierInfo(barrierInfoTest),
          _pageSize(pageSize),
          _taskQueueTypeMap(taskQueueTypeMap),
          _barrierFifoDepth(_barrierFifoDepth),
          _log(log) {
    _pageCount = _barrierInfo.getNumOfBarrierOps() / _pageSize;
    if (_barrierInfo.getNumOfBarrierOps() % _pageSize) {
        _pageCount++;
    }

    initializeTaskToPageAssignment();
    initializeBoundaryTasksData();
}

void vpux::VPURT::BarrierPagesSplitHandler::reconfigureBarrierFifoDepth(size_t barrierFifoDepth) {
    _barrierFifoDepth = barrierFifoDepth;
}

// Configure the barrier page split handler for assignment of barriers and tasks to pages
void vpux::VPURT::BarrierPagesSplitHandler::initializeForAssignment(mlir::func::FuncOp func) {
    _taskQueueTypeMap = VPURT::getTaskOpQueues(func, _barrierInfo);
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
    readTaskPageAssignmentFromIr();
    readBarrierPageAssignmentFromIr();

    initializeBoundaryTasksData();
}

void vpux::VPURT::BarrierPagesSplitHandler::readBarrierPageAssignmentFromIr() {
    _barrierToPageAssignment.resize(_barrierInfo.getNumOfBarrierOps());

    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto barOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(_barrierInfo.getBarrierOpAtIndex(barInd));
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

// Check all tasks and identify boundary tasks.
// A boundary task is one where at least one update barrier belongs to a page
// that is greater than taskPage (indicating cross-page dependency).
// Boundary tasks are later used for legalization purposes.
void vpux::VPURT::BarrierPagesSplitHandler::initializeBoundaryTasksData() {
    _firstAndLastBoundaryTaskForEachPagePerFifo.resize(_pageCount);
    _log.trace("Initializing boundary tasks data");

    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);
        if (updateBars.empty()) {
            continue;
        }

        auto taskPage = _taskPageAssignment[taskInd];
        auto waitBars = _barrierInfo.getWaitBarriers(taskInd);
        VPUX_THROW_WHEN(waitBars.size() > 1, "No support yet for tasks with more than 1 wait barrier, taskInd - {0}",
                        taskInd);

        // Check if all update barriers belong to the same page as the task.
        // If all update barriers are within taskPage, then this is NOT a boundary task
        if (llvm::all_of(updateBars, [&](size_t barInd) {
                return getBarrierPage(barInd) == taskPage;
            })) {
            continue;
        }

        // Get task queue and update boundary tasks per page data
        // Since for each HW FIFO there can be a sequence of boundary tasks
        // store index of first and last one
        auto queueType = _barrierInfo.getTaskQueueType(taskInd);
        if (_firstAndLastBoundaryTaskForEachPagePerFifo[taskPage].find(queueType) ==
            _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage].end()) {
            _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage][queueType] = std::make_pair(taskInd, taskInd);
        } else {
            _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage][queueType].second = taskInd;
        }
    }

    for (size_t pageInd = 0; pageInd < _pageCount - 1; pageInd++) {
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(),
                        "No boundary tasks set for page {0}, pageCount {1}", pageInd, _pageCount);
        _log.nest().trace("Page {0} boundary tasks: {1}", pageInd, getFirstAndLastBoundaryTasksForPage(pageInd));
    }
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

        taskWithWaitBarOnSamePageOpt = _barrierInfo.getPrevTaskOnFifoWithWaitBar(taskWithWaitBarOnSamePageOpt.value());
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
            auto pageBoundaryTask = getLastBoundaryTasksForPage(barProdTaskPage + 1)[0];
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
                    auto prevTaskOpt = _barrierInfo.getPrevTaskOnFifoWithWaitBar(pageBoundaryTask);
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
    auto pageBoundaryTask = getLastBoundaryTasksForPage(taskPage + 1)[0];
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
                _barrierInfo.getPrevTaskOnFifoWithWaitBar(pageBoundaryTaskWithWaitBarOpt.value());

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
    auto taskABlock = _barrierInfo.getControlGraphBlockIndex(taskA);
    auto taskBBlock = _barrierInfo.getControlGraphBlockIndex(taskB);
    // If taskB is on different block than taskA then by definition they are dependant
    if (taskABlock != taskBBlock) {
        return true;
    }

    // If taskA and taskB are on the same HW FIFO and taskA has smaller inder than taskB
    // then dependency exists
    auto taskAQueueType = _barrierInfo.getTaskQueueType(taskA);
    auto taskBQueueType = _barrierInfo.getTaskQueueType(taskB);
    if (taskAQueueType == taskBQueueType && taskA < taskB) {
        return true;
    }

    if (!_blockIdxOfTaskControlMap.has_value() || _blockIdxOfTaskControlMap.value() != taskABlock) {
        _blockIdxOfTaskControlMap = taskABlock;
        _taskControlMapAndOffset = _barrierInfo.buildTaskControlMap(taskABlock);
    }
    auto& [taskControlMap, taskControlMapOffset] = _taskControlMapAndOffset;

    // Check if there is required dep
    return _barrierInfo.controlPathExistsBetweenTasksInSameBlock(taskControlMap, taskA - taskControlMapOffset,
                                                                 taskB - taskControlMapOffset, false);
}

// Check if there is a dependency from task to barrier by checking if there is any dependency
// from this task to any producer of this barrier. If yes then there is a guarantee that task
// needs to execute before barrier is produced - there is topological dependency
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromTaskToBarrier(size_t taskInd, size_t barInd) {
    auto barProdTasks = _barrierInfo.getBarrierProducers(barInd);
    return llvm::any_of(barProdTasks, [&](const auto barProdTask) {
        return isDepFromTaskAToTaskB(taskInd, barProdTask);
    });
}

// Check if there is a dependency from barA to barB by checking if there is any dependency
// from barA consumer to barB producer
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromBarAToBarB(size_t barA, size_t barB) {
    if (barA > barB) {
        return false;
    }

    auto barAConsumers = _barrierInfo.getBarrierConsumers(barA);
    auto barBProducers = _barrierInfo.getBarrierProducers(barB);

    for (auto barAConsumer : barAConsumers) {
        for (auto barBProducer : barBProducers) {
            if (isDepFromTaskAToTaskB(barAConsumer, barBProducer)) {
                return true;
            }
        }
    }
    return false;
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
            auto prevTaskOpt = _barrierInfo.getPrevTaskOnFifoWithWaitBar(taskDst);
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
    auto latestBarriersProdTaskInd = [&](const auto& barInds) {
        size_t latestTaskInd = 0;
        llvm::for_each(barInds, [&](auto barInd) {
            latestTaskInd = std::max(latestTaskInd, _barrierInfo.getBarrierLatestProducer(barInd));
        });

        return latestTaskInd;
    };

    auto earliestBarriersConsTaskInd = [&](const auto& barInds) {
        size_t earliestTaskInd = std::numeric_limits<size_t>::max();
        llvm::for_each(barInds, [&](auto barInd) {
            earliestTaskInd = std::min(earliestTaskInd, _barrierInfo.getBarrierEarliestConsumer(barInd));
        });

        return earliestTaskInd;
    };

    auto startPoint = latestBarriersProdTaskInd(pageStartBars);
    auto endPoint = earliestBarriersConsTaskInd(pageEndBars);

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
            auto newPageStartBar = *std::min_element(syncTaskUpdateBars.begin(), syncTaskUpdateBars.end());

            pageStartBars.clear();
            pageStartTasks.clear();
            pageStartBars.insert(newPageStartBar);
            pageStartTasks.insert(syncTaskOpt.value());
        } else {
            // Use control block sync point as page end task
            _log.trace("Control block boundary: {0} and {1}. Use sync task {2} as endpoint for legalization",
                       barProgDmaBlockInd, pageEndBarBlockInd, syncTaskOpt.value());

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

// Check if any pageStartBar depends on any pageEndBar. Such pageStartBar cannot be used as a starting point
// for a barrier programming DMA which will also produce into pageEndBars. This function will identify such barrier,
// remove it from pageStartBars set and add necessary dependency from boundary task to other, valid pageStartBar
void vpux::VPURT::BarrierPagesSplitHandler::removePageStartBarsDependingOnPageEndBars(
        BarrierInfo::TaskSet& pageStartTasks, BarrierInfo::TaskSet& pageStartBars, BarrierInfo::TaskSet& pageEndBars) {
    BarrierInfo::TaskSet pageStartBarsToLegalize;
    for (auto pageStartBar : pageStartBars) {
        for (auto pageEndBar : pageEndBars) {
            if (!isDepFromBarAToBarB(pageEndBar, pageStartBar)) {
                continue;
            }
            pageStartBarsToLegalize.insert(pageStartBar);
            break;
        }
    }

    if (!pageStartBarsToLegalize.empty()) {
        llvm::for_each(pageStartBarsToLegalize, [&](auto pageStartBarToLegalize) {
            pageStartBars.erase(pageStartBarToLegalize);
        });

        VPUX_THROW_WHEN(pageStartBars.empty(), "No wait barriers for barrier DMA identified");

        // Pick some other pageStartBar that will be used for legalization
        auto startBarToUseForLegalization = *std::max_element(pageStartBars.begin(), pageStartBars.end());

        for (auto pageStartBarToLegalize : pageStartBarsToLegalize) {
            // Legalize by identifying boundary tasks that were producers of pageStartBarToLegalize
            // and make them update other pageStartBar - startBarToUseForLegalization
            _log.trace("Legalizing page start bar {0} which cannot be used as wait barrier for DMA",
                       pageStartBarToLegalize);

            // Find all boundary tasks that is producers of pageStartBarToLegalize
            auto startBarProducersToBeLegalized =
                    to_small_vector(_barrierInfo.getBarrierProducers(pageStartBarToLegalize));

            // Leave only producers which are boundary tasks
            startBarProducersToBeLegalized.erase(llvm::remove_if(startBarProducersToBeLegalized,
                                                                 [&](auto taskInd) {
                                                                     return llvm::find(pageStartTasks, taskInd) ==
                                                                            pageStartTasks.end();
                                                                 }),
                                                 startBarProducersToBeLegalized.end());

            for (auto startBarProducerToBeLegalized : startBarProducersToBeLegalized) {
                _barrierInfo.addProducer(startBarToUseForLegalization, startBarProducerToBeLegalized);
                _log.trace("Add producer {0} to barrier {1}", startBarProducerToBeLegalized,
                           startBarToUseForLegalization);
            }
        }
    }
}

// Get boundary tasks (last on each HW FIFO) from pageInd-1 to pageInd and get barriers they update
// This information will be used as starting point for a page - mark the start
// of barrier DMA
void vpux::VPURT::BarrierPagesSplitHandler::getPageStartTasksAndBars(size_t pageInd,
                                                                     BarrierInfo::TaskSet& pageStartTasks,
                                                                     BarrierInfo::TaskSet& pageStartBars) {
    for (auto& [_, firstLastTaskInd] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1]) {
        auto pageStartTask = firstLastTaskInd.second;
        auto taskUpdateBarsVec = to_small_vector(_barrierInfo.getUpdateBarriers(pageStartTask));

        // Remove barriers that are not on this page - they can be from previous page
        taskUpdateBarsVec.erase(llvm::remove_if(taskUpdateBarsVec,
                                                [&](auto barInd) {
                                                    return getBarrierPage(barInd) != pageInd;
                                                }),
                                taskUpdateBarsVec.end());

        // If task updates multiple barriers, pick only one with smallest index
        // No need to use more barriers as 1 barrier is enough to know that boundary task have finished
        auto startBar = *std::min_element(taskUpdateBarsVec.begin(), taskUpdateBarsVec.end());
        pageStartBars.insert(startBar);
        pageStartTasks.insert(pageStartTask);
        _log.trace("Page start task {0}, start bart {1}", pageStartTask, startBar);
    }
}

// Get boundary tasks from (first on each HW FIFO) pageInd to pageInd+1 and get barriers they wait on
// Get also remaining wait barriers for other boundary tasks on each FIFO for this page
// This information will be used as end point for a page - mark the end
// of barrier DMA
void vpux::VPURT::BarrierPagesSplitHandler::getPageEndTasksAndBars(size_t pageInd, BarrierInfo::TaskSet& pageEndTasks,
                                                                   BarrierInfo::TaskSet& pageEndBars,
                                                                   BarrierInfo::TaskSet& pageEndAllBars) {
    for (auto& [_, firstLastTaskInd] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd]) {
        auto pageEndTask = firstLastTaskInd.first;
        auto taskWaitBars = _barrierInfo.getWaitBarriers(pageEndTask);

        if (taskWaitBars.empty()) {
            auto taskWithWaitBarOnSamePageOpt = _barrierInfo.getPrevTaskOnFifoWithWaitBar(pageEndTask);
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
        _log.trace("Page end task {0}, end bar {1}", pageEndTask, endBar);

        // Get all wait barriers that are used by boundary tasks
        // Scan whole range of boundary task on each FIFO to get all wait barriers
        std::optional<size_t> currentTaskOpt = pageEndTask;
        auto lastTask = firstLastTaskInd.second;
        while (currentTaskOpt.has_value() && currentTaskOpt.value() <= lastTask) {
            auto taskWaitBars = _barrierInfo.getWaitBarriers(currentTaskOpt.value());
            _log.trace("Get page end bars from task {0}", currentTaskOpt.value());
            pageEndAllBars.insert(taskWaitBars.begin(), taskWaitBars.end());
            currentTaskOpt = _barrierInfo.getNextTaskOnFifo(currentTaskOpt.value());
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

    auto setBarProgDmaData = [&](size_t pageInd, BarrierInfo::TaskSet& pageStartBars,
                                 BarrierInfo::TaskSet& pageEndBars) {
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

        // ----------------------------------------
        // Case 1: No common start and end barriers
        // ----------------------------------------
        if (commonStartEndBars.empty()) {
            _log.trace("No common start and end barriers");

            // Check if any pageStartBar depends on any pageEndBar. In that case it cannot be used
            // as wait barrier for barrier DMA
            removePageStartBarsDependingOnPageEndBars(pageStartTasks, pageStartBars, pageEndBars);
            setBarProgDmaData(pageInd, pageStartBars, pageEndBars);
            _log = _log.unnest();
            continue;
        }

        _log.trace("Legalizing common start and end barriers - count {0}", commonStartEndBars.size());

        // Common barriers need to be legalized and related task dependencies updated
        // If there is already other start barrier not part of common barrier, make all common barriers
        // to be end barriers

        auto pageStartOnlyBars = llvm::set_difference(pageStartBars, commonStartEndBars);

        // ------------------------------------------------------------------
        // Case 2: There are barriers which can be treated as start barriers
        // ------------------------------------------------------------------
        // but those which are common, they need to be legalized
        if (!pageStartOnlyBars.empty()) {
            // There are barriers which are start only barriers, but some need to be legalized
            // as they are also used as end barriers
            _log.trace("There is at least one start barrier");

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

            // Check if any pageStartBar depends on any pageEndBar. In that case it cannot be used
            // as wait barrier for barrier DMA
            removePageStartBarsDependingOnPageEndBars(pageStartTasks, pageStartOnlyBars, pageEndBars);

            // Pick one of start only barriers to be used to legalize those boundary tasks
            // TODO: Maybe some heuristic could be used to pick the best barrier for a task?
            auto startBarForLegalization = *std::max_element(pageStartOnlyBars.begin(), pageStartOnlyBars.end());

            for (auto commonStartEndBarBoundaryTaskProducer : commonStartEndBarBoundaryTaskProducers) {
                _barrierInfo.addProducer(startBarForLegalization, commonStartEndBarBoundaryTaskProducer);
                _log.trace("Add producer {0} to barrier {1}", commonStartEndBarBoundaryTaskProducer,
                           startBarForLegalization);
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
            _log.trace("Remove consumer {0} from barrier {1}", startBarConsumer, newStartBar);
            _barrierInfo.addConsumer(endBar, startBarConsumer);
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
                _log.trace("Add producer {0} to barrier {1}", endBarConsumer, updBarToLegalize);
                _barrierInfo.removeProducer(updBarToLegalize, startBarConsumer);
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
        }

        _log.trace("Legalization completed");
        pageStartBars.clear();
        pageStartBars.insert(newStartBar);
        setBarProgDmaData(pageInd, pageStartBars, pageEndBars);

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
