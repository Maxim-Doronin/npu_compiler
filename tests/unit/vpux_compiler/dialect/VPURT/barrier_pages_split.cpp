//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/core/barrier_info.hpp"

#include <gtest/gtest.h>

using namespace vpux;

namespace {

using BarrierPagesSplitTests = ::testing::Test;

/**
 * HW FIFO (DMA): t0 t1 t3
 * HW FIFO (DPU): t2 t4 t5 t6
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1  t2
 *          |  \ |
 *          b1  \|
 *          |    |
 * ------   t3   |
 *          |    b2
 *          |    |
 * Page1    |    t4
 *            \  |
 *              b3
 *               |
 * ------       t5
 *               |
 *              b4
 * Page2         |
 *              t6
 *               |
 *              b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphSimple() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {4},     // task 5
            {5}      // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {4},     // task 5
            {5}      // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckSplitForGraphSimple) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphSimple();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    ASSERT_TRUE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());
    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);

    ASSERT_EQ(page0BoundaryTasks.size(), 3);
    EXPECT_EQ(page0BoundaryTasks[0], 1);
    EXPECT_EQ(page0BoundaryTasks[1], 2);
    EXPECT_EQ(page0BoundaryTasks[2], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2
 * HW FIFO (DPU): t1 t3 t4 t5 t6
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1  t2  <- Long dep: t2->b4. Need to be legalized to t2->b3
 *          |    |
 *          b1   |
 *          |    |
 * ------   t3   |
 *          |    |
 *          b2   |
 * Page1    |    |
 *          t4   |
 *          |    |
 *          b3   |
 *          |    |
 * ------   t5   |
 *           \  /
 *            b4
 * Page2      |
 *            t6
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithLongDepOnTaskUpdateBarrierSide() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {4},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {3},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskUpdateBarrierSide) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithLongDepOnTaskUpdateBarrierSide();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 2 that requires legalization
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 2);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t7
 * HW FIFO (DPU): t1 t3 t4 t5 t6
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1   t2   <-  t2->b4 need to be legalized by connecting t2 to b3
 *          |      |
 *          b1     |
 *          | \    |
 * ------   t3 \   |
 *          |   |  |
 *          b2  |  |
 * Page1    |   |  |
 *          t4  |  |
 *          |   |  |
 *          b3  |  |
 *          |   |  |
 *          t5  |  |
 *             /   |
 *            /    |
 *           /     |
 * ------   t6    /    <- b1->t6 needs to be removed
 *           \   /
 *            b4
 * Page2      |
 *            t7
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphWithLongDepOnTaskUpdateBarrierSideBoundaryTaskDifferentBarrier() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {4},  // task 2
            {2},  // task 3
            {3},  // task 4
            {},   // task 5
            {4},  // task 6
            {5}   // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {1},  // task 6
            {4}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 7};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {3},  // task 2
            {2},  // task 3
            {3},  // task 4
            {},   // task 5
            {4},  // task 6
            {5}   // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {},   // task 6
            {4}   // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskUpdateBarrierSideBoundaryTaskDifferentBarrier) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskUpdateBarrierSideBoundaryTaskDifferentBarrier();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 6);

    // Check that it is taskInd = 2 that requires legalization
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 2);
    EXPECT_EQ(tasksToLegalize[0], 2);
    EXPECT_EQ(tasksToLegalize[1], 6);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t5
 * HW FIFO (DPU): t1 t2 t3 t4
 *
 * ------     t0
 *            |
 *            b0   <- legalize b0->t4 dep - remove it
 *           /  \
 * Page0    t1   |
 *          |    |
 *          b1   |
 *          |    |
 * ------   t2   |
 *          |    |
 *          b2   |
 * Page1    |    |
 *          t3   |
 *         /    /
 *        |    /
 *        |  t4
 *        \  |
 *          b3
 *          |
 * ------   t5
 *  Page2   |
 *          b4
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithLongDepOnTaskWaitBarrierSide() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {0},  // task 4
            {3}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {},   // task 4
            {3}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSide) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithLongDepOnTaskWaitBarrierSide();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 1);
    EXPECT_EQ(page0BoundaryTasks[0], 2);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 4 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 4);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t1 t5
 * HW FIFO (DPU): t2 t3 t4
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t2   t1
 *          |    |
 *          |    b1   <- b1->t4 dependency needs to be legalize by creating
 *          |    |       t1->b2 dependency
 * ------   |    |
 *          |    |
 *          b2   |
 * Page1    |    |
 *          t3   |
 *         /    /
 *        |    /
 *        |  t4
 *        \  |
 *          b3
 *          |
 * ------   t5
 *  Page2   |
 *          b4
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphWithLongDepOnTaskWaitBarrierSideWithBoundaryTaskBarrierShared() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {2},  // task 3
            {1},  // task 4
            {3}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {4}      // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {2},  // task 3
            {},   // task 4
            {3}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSideWithBoundaryTaskBarrierShared) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskWaitBarrierSideWithBoundaryTaskBarrierShared();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 1);
    EXPECT_EQ(page0BoundaryTasks[0], 2);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 4 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 4);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2
 * HW FIFO (DPU): t1 t3 t4 t5 t6 t7
 *
 * ------   t0
 *          |
 *          b0
 *          |  \
 * Page0    t1   t2
 *          |    |
 *          b1   |
 *          |    |
 *          t3   |
 *          |    |
 * ------   t4   |
 *          |    |
 * Page1    |    |
 *          |    b2  <- b2->t7 dep needs to be legalized by connecting t2->b3
 *          b3   |
 *          |    |
 * ------   t5   |
 *          |    |
 *          b4   |
 * Page2    |    |
 *          t6   /
 *         /    /
 *        |    /
 *        |  t7
 *        \  |
 *          b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPage() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {},   // task 3
            {3},  // task 4
            {4},  // task 5
            {},   // task 6
            {5}   // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {},   // task 4
            {3},  // task 5
            {4},  // task 6
            {2}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 4, 5, 6, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {},      // task 3
            {3},     // task 4
            {4},     // task 5
            {},      // task 6
            {5}      // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {},   // task 4
            {3},  // task 5
            {4},  // task 6
            {}    // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPage) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPage();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 4);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 4 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 7);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0
 * HW FIFO (DPU): t1 t2 t3 t4 t5 t6
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |  \
 *          b1  \
 *          |    |
 *          t2   |
 *          |    |
 * ------   t3   |
 *          |    |
 * Page1    |    |
 *          |    b2  -> b2->t6 needs to be legalized by removing it
 *          b3   |
 *          |    |
 * ------   t4   |
 *          |    |
 *          b4   |
 * Page2    |    |
 *          t5   /
 *         /    /
 *        |    /
 *        |  t6
 *        \  |
 *          b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPageOnSameFifo() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {},      // task 2
            {3},     // task 3
            {4},     // task 4
            {},      // task 5
            {5}      // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {},   // task 3
            {3},  // task 4
            {4},  // task 5
            {2}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 3, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {},      // task 2
            {3},     // task 3
            {4},     // task 4
            {},      // task 5
            {5}      // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {},   // task 3
            {3},  // task 4
            {4},  // task 5
            {}    // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierOnSameFifo) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPageOnSameFifo();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 1);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 4);

    // Check that it is taskInd = 4 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 6);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6 t8
 * HW FIFO (DPU): t1 t3 t5 t7
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1   |
 *          |    |
 *          b1   |
 *          |    |
 * ------   t3   t2
 *          |    |
 * Page1    b3   b2
 *          |    |
 * ------   t5   t4  -> Create t2->b3 and t3->b2 deps
 *          |    |
 * Page2    b5   b4
 *          |    |
 * ------   t7   t6  -> Create t4->b5 and t5->b4 deps
 *           \  /
 *            b6
 * Page3      |
 *            t8
 *            |
 *            b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithBoundaryTaskDepsToLegalize() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7}   // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {2, 3},  // task 3
            {4, 5},  // task 4
            {4, 5},  // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeBoundaryTaskDeps) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithBoundaryTaskDepsToLegalize();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    EXPECT_FALSE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto boundaryTaskPairsMissingDepInBetween = barrierPagesSplitHandlerTest.getBoundaryTaskPairsMissingDepInBetween();

    // Expected missing deps in graph
    //  3 -> 4
    //  2 -> 5
    //  5 -> 6
    //  4 -> 7
    EXPECT_EQ(boundaryTaskPairsMissingDepInBetween.size(), 4);

    auto it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                        std::make_pair<size_t, size_t>(3, 4));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(2, 5));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(5, 6));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(4, 7));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    barrierPagesSplitHandlerTest.legalizeDepsForBoundaryTasks(boundaryTaskPairsMissingDepInBetween);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6 t8
 * HW FIFO (DPU): t1 t3 t5 t7
 *
 * ------      t0
 *             |
 *             b0
 *           /    \
 * Page0    t1     |
 *          |      |
 *          b1     |
 *          |      |
 * ------   t3    t2  <- Create t2->b3 and t3->b2 deps
 *          |    / |
 * Page1    b3  |  b2
 *          |   |  |
 * ------   t5  |  t4  <- Create t4->b5 and t5->b4 deps
 *          |  /   |
 * Page2    b5     b4
 *          |      |
 * ------   t7     t6
 *           \    /
 *             b6
 * Page3       |
 *             t8
 *             |
 *             b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithBoundaryTasksWithMultipleUpdateBars() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 5},  // task 2
            {3},     // task 3
            {4},     // task 4
            {5},     // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},        // task 0
            {1},        // task 1
            {2, 3, 5},  // task 2
            {2, 3},     // task 3
            {4, 5},     // task 4
            {4, 5},     // task 5
            {6},        // task 6
            {6},        // task 7
            {7}         // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeBoundaryTaskDepsWithMultipleUpdateBars) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithBoundaryTasksWithMultipleUpdateBars();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    EXPECT_FALSE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 2);
    EXPECT_EQ(page1BoundaryTasks[0], 4);
    EXPECT_EQ(page1BoundaryTasks[1], 5);

    auto page2BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(2);
    ASSERT_EQ(page2BoundaryTasks.size(), 2);
    EXPECT_EQ(page2BoundaryTasks[0], 6);
    EXPECT_EQ(page2BoundaryTasks[1], 7);

    auto boundaryTaskPairsMissingDepInBetween = barrierPagesSplitHandlerTest.getBoundaryTaskPairsMissingDepInBetween();

    // Expected missing deps in graph
    //  3 -> 4
    //  2 -> 5
    //  5 -> 6
    //  4 -> 7
    EXPECT_EQ(boundaryTaskPairsMissingDepInBetween.size(), 4);

    auto it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                        std::make_pair<size_t, size_t>(3, 4));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(2, 5));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(5, 6));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(4, 7));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    barrierPagesSplitHandlerTest.legalizeDepsForBoundaryTasks(boundaryTaskPairsMissingDepInBetween);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t2 t3 t6 t7 t8
 * HW FIFO (DPU): t0 t1 t4 t5
 *
 * ------   t0
 *            \
 *             b0
 *           /  |
 * Page0    t1  t2
 *          |     \
 *          b1     \
 *            \     |
 * ------       t3  |
 *            /    /
 * Page1    b3    b2
 *        /     /
 *        |    /
 *        | t4
 *        |    \
 *        \     \
 * ------   t5  |
 *          |   |
 * Page2    b5  b4
 *          |   |
 *          |   t6
 *           \    \
 *            \    \
 * ------       t7  |
 *              |  /
 *              b6
 * Page3        |
 *              t8
 *              |
 *              b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithMultipleBoundaryTasksOnSameFifo() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7}   // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {2, 3, 6, 7, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {0, 1, 4, 5};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {2, 3},  // task 3
            {4},     // task 4
            {4, 5},  // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeMultipleBoundaryTasksOnSameFifo) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithMultipleBoundaryTasksOnSameFifo();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    EXPECT_FALSE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 2);
    EXPECT_EQ(page1BoundaryTasks[0], 4);
    EXPECT_EQ(page1BoundaryTasks[1], 5);

    auto page2BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(2);
    ASSERT_EQ(page2BoundaryTasks.size(), 2);
    EXPECT_EQ(page2BoundaryTasks[0], 6);
    EXPECT_EQ(page2BoundaryTasks[1], 7);

    auto boundaryTaskPairsMissingDepInBetween = barrierPagesSplitHandlerTest.getBoundaryTaskPairsMissingDepInBetween();

    barrierPagesSplitHandlerTest.legalizeDepsForBoundaryTasks(boundaryTaskPairsMissingDepInBetween);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t5
 * HW FIFO (DPU): t1 t2 t3 t4
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |
 *          b1
 *          |
 * ------   t2
 *          |
 *          b2
 * Page1    |
 *          t3
 *          |
 *          b3
 *          |
 * ------   t4
 *          |
 *          b4
 * Page2    |
 *          t5
 *          |
 *          b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphGetBarrierDmaLocation() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetBarrierDmaLocation) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphGetBarrierDmaLocation();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();
    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 2);
}

/**
 * HW FIFO (DMA): t0 t1 t2 t3 t5
 * HW FIFO (DPU): t4
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |
 *          b1
 *          |
 * ------   t2
 *          |
 *          b2
 *          |
 *  Page1   t3      <- Remove t3->b4 and change it to t4->b4. Slot for inserting barrier DMA
 *          |  \       prepared between b2 and b3
 *          |   b3
 *          |   |
 * ------   |   t4
 *          |   |
 *          b4  |
 * Page2    |   |
 *          t5  |
 *           \ /
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereStartBarIsEndBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3, 4},  // task 3
            {5},     // task 4
            {5}      // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 2, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4, 5},  // task 4
            {5}      // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereStartBarIsEndBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphToLegalizeForBarrierDmaWhereStartBarIsEndBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 2);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t8
 * HW FIFO (DPU): t1 t3 t5 t6 t7
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1   \
 *          |     |
 *          b1    |
 *          |     |
 * ------   t3    t2
 *          |  /\ |
 * Page1    b3    b2    <- b2->t4 dep to be removed and replaced with b3->t4.  Slot for inserting barrier DMA
 *          |     |        prepared between b2 and b3
 * ------   t5    t4
 *          |  \/ |
 * Page2    b5    b4    <- b4->t6 dep to be removed and replaced with b5->t6.  Slot for inserting barrier DMA
 *          |     |        prepared between b4 and b5
 * ------   t7    t6
 *           \  /
 *            b6
 * Page3      |
 *            t8
 *            |
 *            b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereAllStartBarsAreEndBars() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {2, 3},  // task 3
            {4, 5},  // task 4
            {4, 5},  // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 6, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {2, 3},  // task 3
            {4, 5},  // task 4
            {4, 5},  // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {3},  // task 4
            {3},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereAllStartBarsAreEndBars) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereAllStartBarsAreEndBars();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 3);

    auto barProgDmaPosPage2 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(2);

    ASSERT_TRUE(barProgDmaPosPage2.valid);
    ASSERT_EQ(barProgDmaPosPage2.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage2.waitBars[0], 4);
    ASSERT_EQ(barProgDmaPosPage2.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage2.updateBars[0], 5);
    EXPECT_EQ(barProgDmaPosPage2.insertAfter, 5);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6
 * HW FIFO (DPU): t1 t3 t5 t7
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *            t1
 *            |
 * Page0      b1
 *            |
 *            t2
 *            |
 *            b2
 *           /   \
 * ------   t3    t4
 *          |  \   | \
 *          b3    b4  |   <- remove t4->b4 dependency and add t4->b3.
 *          |      |  |      Slot for inserting barrier DMA prepared between b3 and b4&b5
 *  Page1   t5     |  |
 *          |      |  |
 *          b5 <------
 *          |      |
 * ------   t6    t7
 *          |   /
 *   Page2  b6
 *
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereNotAllStartBarsAreEndBars() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3, 4},  // task 3
            {4, 5},  // task 4
            {5},     // task 5
            {6},     // task 6
            {6}      // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {5},  // task 6
            {4}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 7};

    size_t pageSize = 3;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},        // task 0
            {1},        // task 1
            {2},        // task 2
            {3, 4},     // task 3
            {3, 4, 5},  // task 4
            {5},        // task 5
            {6},        // task 6
            {6}         // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {5},  // task 6
            {4}   // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereNotAllStartBarsAreEndBars) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereNotAllStartBarsAreEndBars();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 3);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 2);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 4);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[1], 5);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6 t8
 * HW FIFO (DPU): t1 t3 t5 t7
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *            t1
 *            |
 * Page0      b1
 *            |
 *            t2
 *            |
 *            b2
 *           /   \
 * ------   t3    t4
 *          |   /  |
 *          b3     |
 *          |  \   |
 *  Page1   |   t5 |
 *          |    \ |
 *          |      b4
 *          |      |
 *          |      t6
 *          |      |
 *          |      b5   <- create b5->t7 dependency. Barrier DMA to wait on b3 and update b5
 *          |      |
 * ------   t7    t8
 *          |   /
 *   Page2  b6
 *
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereStartBarDependsOnEndBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {3, 4},  // task 4
            {4},     // task 5
            {5},     // task 6
            {6},     // task 7
            {6}      // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {3},  // task 7
            {5}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 7};

    size_t pageSize = 3;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {3, 4},  // task 4
            {4},     // task 5
            {5},     // task 6
            {6},     // task 7
            {6}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {5}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereStartBarDependsOnEndBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereStartBarDependsOnEndBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 3);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 5);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t1 t2 t3 t5 t9
 * HW FIFO (DPU): t4 t6 t7 t8 t10
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *            t1
 *            |
 * Page0      b1
 *            |
 *            t2
 *            |
 *            b2
 *            |
 *            t3
 *            |
 *            b3
 *           /   \
 * ------   t4    t5   <- Create t5->b4 dep
 *          |      |      Slot for inserting barrier DMA prepared between b4 and b4&b7
 *          b4     |
 *          |      |
 *  Page1   t6     |
 *          |      |
 *          b5     |
 *          | \    |
 *          |  t7  |
 *          |   \  /
 *          |    b6
 *          |    |
 *          |    t8
 *          |    |
 *          |    b7
 *          |    |
 * ------   t9   t10
 *          |   /
 *   Page2  b8
 *
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphToLegalizeForBarrierDmaWhereOneOfWaitBarriersWouldDependOnUpdate() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {6},  // task 5
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
            {8},  // task 9
            {8}   // task 10
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6},  // task 8
            {5},  // task 9
            {7}   // task 10
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 2, 3, 5, 9};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {4, 6, 7, 8, 10};

    size_t pageSize = 4;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4},     // task 4
            {4, 6},  // task 5
            {5},     // task 6
            {6},     // task 7
            {7},     // task 8
            {8},     // task 9
            {8}      // task 10
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6},  // task 8
            {5},  // task 9
            {7}   // task 10
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereOneOfWaitBarriersWouldDependOnUpdate) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereOneOfWaitBarriersWouldDependOnUpdate();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 4);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 2);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 5);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[1], 7);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 5);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t5 t6 t9
 * HW FIFO (DMA1): t1 t8
 * HW FIFO (DPU): t2 t3 t4 t7
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page1     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |
 * ------     t4  <- Page1 has no boundary tasks of DMA0 and DMA1 type
 *            |      both need to be inserted
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 *           /   \
 * ------   t6    t7   <- Page2 has boundary tasks only of DMA0 type
 *          |  /   |      DMA1 type dummy DMA needs to be inserted
 *  Page3   b6     b7
 *          |      |
 * ------   t8   t9
 *          |   /
 *  Page4   b8
 *
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToInsertDummyDma() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4},     // task 4
            {5},     // task 5
            {6},     // task 6
            {6, 7},  // task 7
            {8},     // task 8
            {8}      // task 9
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6},  // task 8
            {7}   // task 9
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 5, 6, 9};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 3, 4, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4},     // task 4
            {5},     // task 5
            {6},     // task 6
            {6, 7},  // task 7
            {8},     // task 8
            {8}      // task 9
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6},  // task 8
            {7}   // task 9
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionData) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphToInsertDummyDma();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};

    ASSERT_EQ(dummyDmaInsertionData.size(), 3);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType0);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 3);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].updateBars[0], 5);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 3);

    EXPECT_EQ(dummyDmaInsertionData[1].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[1].queueType, dmaType1);
    ASSERT_EQ(dummyDmaInsertionData[1].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[1].waitBars[0], 3);
    ASSERT_EQ(dummyDmaInsertionData[1].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[1].updateBars[0], 5);
    EXPECT_EQ(dummyDmaInsertionData[1].insertAfter, 3);

    EXPECT_EQ(dummyDmaInsertionData[2].pageInd, 2);
    EXPECT_EQ(dummyDmaInsertionData[2].queueType, dmaType1);
    ASSERT_EQ(dummyDmaInsertionData[2].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[2].waitBars[0], 5);
    ASSERT_EQ(dummyDmaInsertionData[2].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[2].updateBars[0], 7);
    EXPECT_EQ(dummyDmaInsertionData[2].insertAfter, 5);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t5
 * HW FIFO (DMA1): t1
 * HW FIFO (DPU): t2 t3 t4
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page1     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |
 * ------     t4 (sync)  <- Page1 has no boundary tasks of DMA0 type
 *            |
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToInsertDummyDmaWithSyncPoint() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {4};

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 5};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionDataWithSyncPoint) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphToInsertDummyDmaWithSyncPoint();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};

    ASSERT_EQ(dummyDmaInsertionData.size(), 1);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType0);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 2);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].updateBars[0], 3);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 2);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t6
 * HW FIFO (DMA1): t1 t3
 * HW FIFO (DPU): t2 t4 t5
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |  \
 * ------     t2  t3
 *            |   |
 *            b2  |
 *            |   |
 *  Page1     t4  |
 *            |  /
 *            b3
 *            |
 * ------     t5 (sync)  <- Page1 has no boundary tasks of DMA0 type
 *            |
 *            b4
 *            |
 *  Page2     t6
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToInsertDummyDmaWithSyncPointAndCommonStartEndBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {5};

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionDataWithSyncPointAndCommonStartEndBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToInsertDummyDmaWithSyncPointAndCommonStartEndBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};

    ASSERT_EQ(dummyDmaInsertionData.size(), 1);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType0);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 2);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].updateBars[0], 3);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 3);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t6
 * HW FIFO (DMA1): t1 t3
 * HW FIFO (DPU): t2 t4 t5
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |  \
 * ------     t2  t3
 *            |  /
 *            b2
 *            |
 *  Page1     t4
 *            |
 *            b3
 *            |
 * ------     t5
 *            |
 *            b4
 *            |
 *  Page2     t6
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueAtBootstrapAndStartBarrier() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5};

    size_t pageSize = 2;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueAtBootstrapAndStartBarrier) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig] =
            graphToCheckEnqueueAtBootstrapAndStartBarrier();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::VPU::ExecutorKind> executorEnqAtBootstrap{vpux::VPU::ExecutorKind::DMA_NN};
    auto enqueueBarVec = barrierPagesSplitHandlerTest.prepareEnqueueDmaBarForFullWlm(executorEnqAtBootstrap);

    ASSERT_EQ(enqueueBarVec.size(), 7);

    // DMA tasks should be enqueued at bootstrap
    EXPECT_FALSE(enqueueBarVec[0].has_value());
    EXPECT_FALSE(enqueueBarVec[1].has_value());
    EXPECT_FALSE(enqueueBarVec[3].has_value());
    EXPECT_FALSE(enqueueBarVec[6].has_value());

    // DPU tasks enqueued at start barrier
    EXPECT_TRUE(enqueueBarVec[2].has_value());
    EXPECT_EQ(enqueueBarVec[2].value(), 0);
    EXPECT_TRUE(enqueueBarVec[4].has_value());
    EXPECT_EQ(enqueueBarVec[4].value(), 0);
    EXPECT_TRUE(enqueueBarVec[5].has_value());
    EXPECT_EQ(enqueueBarVec[5].value(), 0);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4
 * HW FIFO (DMA1): t1 t3 t5
 * HW FIFO (DPU): t6
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |
 * ------     t4
 *            |
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 *            |
 * ------     t6
 *  Page3     |
 *            b6
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, BarrierInfoMaps> graphToCheckEnqueueOfDpuNotAtStartBarrier() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {6};

    size_t pageSize = 2;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1, 2};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuNotAtStartBarrier) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuNotAtStartBarrier();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::VPU::ExecutorKind> executorEnqAtBootstrap{vpux::VPU::ExecutorKind::DMA_NN};
    auto enqueueBarVec = barrierPagesSplitHandlerTest.prepareEnqueueDmaBarForFullWlm(executorEnqAtBootstrap);

    ASSERT_EQ(enqueueBarVec.size(), 7);

    // DMA tasks should be enqueued at bootstrap
    for (size_t taskInd = 0; taskInd < 6; ++taskInd) {
        EXPECT_FALSE(enqueueBarVec[taskInd].has_value());
    }

    // DPU task from Page 2 enqueued at end barrier of Page 1
    ASSERT_TRUE(enqueueBarVec[6].has_value());
    EXPECT_EQ(enqueueBarVec[6].value(), 3);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4 t6
 * HW FIFO (DMA1): t1 t3 t5
 * HW FIFO (DPU): t7 t8
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |
 * ------     t4
 *            |
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 *            |  \
 * ------     t6 t7
 *            |  /
 *            b6
 *  Page3     |
 *            t8
 *            |
 *            b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, BarrierInfoMaps> graphToCheckEnqueueMergeOfDpu() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7}   // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {7, 8};

    size_t pageSize = 2;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1, 2, 3};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7}   // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueMergeOfDpu) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig] = graphToCheckEnqueueMergeOfDpu();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::VPU::ExecutorKind> executorEnqAtBootstrap{vpux::VPU::ExecutorKind::DMA_NN};
    auto enqueueBarVec = barrierPagesSplitHandlerTest.prepareEnqueueDmaBarForFullWlm(executorEnqAtBootstrap);

    ASSERT_EQ(enqueueBarVec.size(), 9);

    // DMA tasks should be enqueued at bootstrap
    for (size_t taskInd = 0; taskInd < 7; ++taskInd) {
        EXPECT_FALSE(enqueueBarVec[taskInd].has_value());
    }

    // DPU task from Page 2 enqueued at end barrier of Page 1
    ASSERT_TRUE(enqueueBarVec[7].has_value());
    EXPECT_EQ(enqueueBarVec[7].value(), 3);

    // DPU task from Page 3 enqueued together with previous DPU
    ASSERT_TRUE(enqueueBarVec[8].has_value());
    EXPECT_EQ(enqueueBarVec[8].value(), enqueueBarVec[7].value());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2
 * HW FIFO (DMA1): t1 t3 t6
 * HW FIFO (SHV(withDPU)): t5
 * HW FIFO (DPU): t7
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |  \
 * ------     t4  t5
 *            |  /
 *            b4
 *            |
 *  Page2     t6
 *            |
 *            b5
 *            |
 * ------     t7
 *  Page3     |
 *            b6
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec, shvTasksWithDpu and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueOfDpuDelayedDueToShvWithDpu() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {4},  // task 5
            {5},  // task 6
            {6}   // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{VPU::ExecutorKind::SHAVE_NN, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {7};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {5};

    size_t pageSize = 2;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1, 2};

    SmallVector<size_t> shvTasksWithDpu = {5};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {4},  // task 5
            {5},  // task 6
            {6}   // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5}   // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuDelayedDueToShvWithDpu) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuDelayedDueToShvWithDpu();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1, shvTasksWithDpu);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::VPU::ExecutorKind> executorEnqAtBootstrap{vpux::VPU::ExecutorKind::DMA_NN};
    auto enqueueBarVec = barrierPagesSplitHandlerTest.prepareEnqueueDmaBarForFullWlm(executorEnqAtBootstrap);

    ASSERT_EQ(enqueueBarVec.size(), 8);

    // SHV task enqueued at startbarrier
    ASSERT_TRUE(enqueueBarVec[5].has_value());
    EXPECT_EQ(enqueueBarVec[5].value(), 0);

    // DPU task from Page 2 has enqueue delayed to be after SHV task which submits DPU
    ASSERT_TRUE(enqueueBarVec[7].has_value());
    EXPECT_EQ(enqueueBarVec[7].value(), 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

}  // namespace
