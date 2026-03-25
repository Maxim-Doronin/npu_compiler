//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/hardware_simulator.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace vpux::VPUIP::scheduling::simulator {
class HardwareSimulatorTester {
public:
    using FIFOType = HardwareSimulator::HardwareFIFO;

    // Helper functions to expose the inner FIFO type. This makes modular testing easier.
    static auto makeFIFO(size_t executorCount) {
        return FIFOType(executorCount);
    }

    static void checkFIFOInvariants(const FIFOType& fifo) {
        for (const auto& jobs : fifo.scheduledOpInfosPerExecutor) {
            for (size_t i = 1; i < jobs.size(); ++i) {
                if (jobs[i - 1]->getCycleEnd() > jobs[i]->getCycleEnd()) {
                    FAIL() << "Jobs are not sorted by cycleEnd!";
                }
            }
        }
    }

    static size_t getEndCycle(const FIFOType& fifo) {
        const auto it = std::max_element(fifo.scheduledOpInfosPerExecutor.begin(),
                                         fifo.scheduledOpInfosPerExecutor.end(), [](const auto& a, const auto& b) {
                                             const auto aEnd = a.empty() ? 0 : a.back()->getCycleEnd();
                                             const auto bEnd = b.empty() ? 0 : b.back()->getCycleEnd();
                                             return aEnd < bEnd;
                                         });
        return it->back()->getCycleEnd();
    }

    static void expectEndCycle(const FIFOType& fifo, size_t expectedEndCycle) {
        const auto endCycle = getEndCycle(fifo);
        EXPECT_EQ(endCycle, expectedEndCycle);
    }
};
}  // namespace vpux::VPUIP::scheduling::simulator

namespace {

using QueueType = VPURT::TaskQueueType;
using HardwareSimulator = vpux::VPUIP::scheduling::simulator::HardwareSimulator;
using HardwareSimulatorTester = vpux::VPUIP::scheduling::simulator::HardwareSimulatorTester;

class MLIR_HardwareSimulator : public MLIR_UnitBase {
public:
    static QueueType getDMAType(uint8_t id = 0) noexcept {
        return QueueType(config::ExecutorKind::DMA_NN, id);
    }

    static QueueType getDPUType() noexcept {
        return QueueType(config::ExecutorKind::DPU, 0);
    }

    static QueueType getNCEType() noexcept {
        return QueueType(config::ExecutorKind::NCE, 0);
    }
};

TEST_F(MLIR_HardwareSimulator, FIFO_API_Checks) {
    EXPECT_DEBUG_DEATH(HardwareSimulatorTester::makeFIFO(0),
                       "Number of executors must be between 1 and 64 \\(inclusive\\)");
    EXPECT_DEBUG_DEATH(HardwareSimulatorTester::makeFIFO(65),
                       "Number of executors must be between 1 and 64 \\(inclusive\\)");

    auto fifo1 = HardwareSimulatorTester::makeFIFO(1);
    EXPECT_THROW(fifo1.findNextAvailableSlot(0), vpux::Exception);
    fifo1.findNextAvailableSlot(1);
    EXPECT_THROW(fifo1.findNextAvailableSlot(2), vpux::Exception);

    auto fifo2 = HardwareSimulatorTester::makeFIFO(2);
    EXPECT_THROW(fifo2.findNextAvailableSlot(0), vpux::Exception);
    fifo2.findNextAvailableSlot(1);
    fifo2.findNextAvailableSlot(2);
    EXPECT_THROW(fifo2.findNextAvailableSlot(3), vpux::Exception);

    auto fifo4 = HardwareSimulatorTester::makeFIFO(4);
    EXPECT_THROW(fifo4.findNextAvailableSlot(0), vpux::Exception);
    fifo4.findNextAvailableSlot(1);
    EXPECT_THROW(fifo4.findNextAvailableSlot(2), vpux::Exception);
    EXPECT_THROW(fifo4.findNextAvailableSlot(3), vpux::Exception);
    fifo4.findNextAvailableSlot(4);
    EXPECT_THROW(fifo4.findNextAvailableSlot(5), vpux::Exception);
}

/// This tests the following scenario:
///                ┌─────────────────────────────────────────────────────────────────────────┐
///    Executor 0  │ A [0-10]  │  B [10-30]   │ C [30-35] │ D [35-43] │ E [43-60] │          │
///                │             │ F [15-30]  │           │           │          │ G [59-64] │
///                └─────────────────────────────────────────────────────────────────────────┘
///    F - overlaps, does not extend end cycle
///    G - overlaps, extends end cycle
TEST_F(MLIR_HardwareSimulator, FIFO_1Executor) {
    auto fifo = HardwareSimulatorTester::makeFIFO(1);

    const std::vector<size_t> cycleCosts{10, 20, 5, 8, 17};

    for (const size_t cycleCost : cycleCosts) {
        auto slot = fifo.findNextAvailableSlot(1);
        fifo.scheduleOp(cycleCost, slot);
    }
    HardwareSimulatorTester::checkFIFOInvariants(fifo);
    HardwareSimulatorTester::expectEndCycle(fifo, 60);

    {
        const auto& jobs = fifo.scheduledOpInfosPerExecutor[0];
        EXPECT_EQ(jobs.size(), 5);
        EXPECT_EQ(jobs[0]->cycleBegin, 0);
        EXPECT_EQ(jobs[1]->cycleBegin, 10);
        EXPECT_EQ(jobs[2]->cycleBegin, 30);
        EXPECT_EQ(jobs[3]->cycleBegin, 35);
        EXPECT_EQ(jobs[4]->cycleBegin, 43);
    }

    // Schedule a job that overlaps and not change the final cycle.
    fifo.scheduleOp(5, {15, 0x1});
    HardwareSimulatorTester::checkFIFOInvariants(fifo);
    HardwareSimulatorTester::expectEndCycle(fifo, 60);

    {
        const auto& jobs = fifo.scheduledOpInfosPerExecutor[0];
        EXPECT_EQ(jobs.size(), 6);
        EXPECT_EQ(jobs[0]->cycleBegin, 0);
        EXPECT_EQ(jobs[1]->cycleBegin, 15);  // overlapping job
        EXPECT_EQ(jobs[2]->cycleBegin, 10);
        EXPECT_EQ(jobs[3]->cycleBegin, 30);
        EXPECT_EQ(jobs[4]->cycleBegin, 35);
        EXPECT_EQ(jobs[5]->cycleBegin, 43);
    }

    // Schedule a job that overlaps and change the final cycle.
    fifo.scheduleOp(5, {59, 0x1});
    HardwareSimulatorTester::checkFIFOInvariants(fifo);
    HardwareSimulatorTester::expectEndCycle(fifo, 64);

    {
        const auto& jobs = fifo.scheduledOpInfosPerExecutor[0];
        EXPECT_EQ(jobs.size(), 7);
        EXPECT_EQ(jobs[0]->cycleBegin, 0);
        EXPECT_EQ(jobs[1]->cycleBegin, 15);  // overlapping job
        EXPECT_EQ(jobs[2]->cycleBegin, 10);
        EXPECT_EQ(jobs[3]->cycleBegin, 30);
        EXPECT_EQ(jobs[4]->cycleBegin, 35);
        EXPECT_EQ(jobs[5]->cycleBegin, 43);
        EXPECT_EQ(jobs[6]->cycleBegin, 59);  // overlapping job
    }
}

/// This tests the following scenario:
///                ┌───────────────────────────────────────────────────────────────┐
///    Executor 0  │ A [0-10]  │  B [10-30]   │ C [30-35] │ │ E [38-55] │          │
///                │             │ F [15-30]  │                                    │
///                ├───────────────────────────────────────────────────────────────┤
///    Executor 1  │           │  B [10-30]   │ D [30-38]   │ E [38-55] │          │
///                │                                                   │ G [54-59] │
///                └───────────────────────────────────────────────────────────────┘
///    F - executor 0, overlaps, does not extend end cycle
///    G - executor 1, overlaps, extends end cycle
TEST_F(MLIR_HardwareSimulator, FIFO_2Executors) {
    auto fifo = HardwareSimulatorTester::makeFIFO(2);

    const std::vector<size_t> cycleCosts{10, 20, 5, 8, 17};
    const std::vector<size_t> requiredExecutors{1, 2, 1, 1, 2};

    for (const auto& [cycleCost, requiredExecutor] : llvm::zip(cycleCosts, requiredExecutors)) {
        const auto slot = fifo.findNextAvailableSlot(requiredExecutor);
        fifo.scheduleOp(cycleCost, slot);
    }
    HardwareSimulatorTester::checkFIFOInvariants(fifo);
    HardwareSimulatorTester::expectEndCycle(fifo, 55);

    {
        const auto& jobs0 = fifo.scheduledOpInfosPerExecutor[0];
        EXPECT_EQ(jobs0.size(), 4);
        EXPECT_EQ(jobs0[0]->cycleBegin, 0);
        EXPECT_EQ(jobs0[1]->cycleBegin, 10);
        EXPECT_EQ(jobs0[2]->cycleBegin, 30);
        EXPECT_EQ(jobs0[3]->cycleBegin, 38);

        const auto& jobs1 = fifo.scheduledOpInfosPerExecutor[1];
        EXPECT_EQ(jobs1.size(), 3);
        EXPECT_EQ(jobs1[0]->cycleBegin, 10);
        EXPECT_EQ(jobs1[1]->cycleBegin, 30);
        EXPECT_EQ(jobs1[2]->cycleBegin, 38);
    }

    // Schedule a job that overlaps but does not change the final cycle.
    fifo.scheduleOp(5, {15, 0x1});
    HardwareSimulatorTester::checkFIFOInvariants(fifo);
    HardwareSimulatorTester::expectEndCycle(fifo, 55);

    {
        const auto& jobs0 = fifo.scheduledOpInfosPerExecutor[0];
        EXPECT_EQ(jobs0.size(), 5);
        EXPECT_EQ(jobs0[0]->cycleBegin, 0);
        EXPECT_EQ(jobs0[1]->cycleBegin, 15);  // overlapping job
        EXPECT_EQ(jobs0[2]->cycleBegin, 10);
        EXPECT_EQ(jobs0[3]->cycleBegin, 30);
        EXPECT_EQ(jobs0[4]->cycleBegin, 38);

        const auto& jobs1 = fifo.scheduledOpInfosPerExecutor[1];
        EXPECT_EQ(jobs1.size(), 3);
        EXPECT_EQ(jobs1[0]->cycleBegin, 10);
        EXPECT_EQ(jobs1[1]->cycleBegin, 30);
        EXPECT_EQ(jobs1[2]->cycleBegin, 38);
    }

    // Schedule a job that overlaps and changes the final cycle.
    fifo.scheduleOp(5, {54, 0x2});
    HardwareSimulatorTester::checkFIFOInvariants(fifo);
    HardwareSimulatorTester::expectEndCycle(fifo, 59);

    {
        const auto& jobs0 = fifo.scheduledOpInfosPerExecutor[0];
        EXPECT_EQ(jobs0.size(), 5);
        EXPECT_EQ(jobs0[0]->cycleBegin, 0);
        EXPECT_EQ(jobs0[1]->cycleBegin, 15);  // overlapping job
        EXPECT_EQ(jobs0[2]->cycleBegin, 10);
        EXPECT_EQ(jobs0[3]->cycleBegin, 30);
        EXPECT_EQ(jobs0[4]->cycleBegin, 38);

        const auto& jobs1 = fifo.scheduledOpInfosPerExecutor[1];
        EXPECT_EQ(jobs1.size(), 4);
        EXPECT_EQ(jobs1[0]->cycleBegin, 10);
        EXPECT_EQ(jobs1[1]->cycleBegin, 30);
        EXPECT_EQ(jobs1[2]->cycleBegin, 38);
        EXPECT_EQ(jobs1[3]->cycleBegin, 54);  // overlapping job
    }
}

/// This tests the following scenario:
///                ┌───────────────────┐
///    Executor 0  │ A [0-2] │ E [2-4] │
///                ├───────────────────┤
///    Executor 1  │ B [0-2] │ F [2-4] │
///                ├───────────────────┤
///    Executor 2  │ C [0-2] │ G [2-4] │
///                ├───────────────────┤
///    Executor 3  │ D [0-2] │ H [2-4] │
///                └───────────────────┘
///    Tests that 8 jobs (each 2 cycles) are tightly packed across 4 executors.
///    All executors should end at cycle 4.
TEST_F(MLIR_HardwareSimulator, FIFO_TightPacking) {
    auto fifo = HardwareSimulatorTester::makeFIFO(4);

    for (size_t i = 0; i < 8; ++i) {
        const auto slot = fifo.findNextAvailableSlot(1);
        fifo.scheduleOp(2, slot);
    }

    for (size_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(!fifo.scheduledOpInfosPerExecutor[i].empty());
        EXPECT_EQ(fifo.scheduledOpInfosPerExecutor[i].back()->getCycleEnd(), 4);
    }
}

/// This tests that the jobs are pushed back correctly when stalls are inserted.
TEST_F(MLIR_HardwareSimulator, FIFO_Stalling) {
    auto fifo = HardwareSimulatorTester::makeFIFO(4);

    for (size_t i = 0; i < 8; ++i) {
        const auto slot = fifo.findNextAvailableSlot(1);
        fifo.scheduleOp(2, slot);
    }

    fifo.insertStall(0, 2);

    HardwareSimulatorTester::checkFIFOInvariants(fifo);

    for (size_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(!fifo.scheduledOpInfosPerExecutor[i].empty());
        EXPECT_EQ(fifo.scheduledOpInfosPerExecutor[i].back()->getCycleEnd(), 6);
    }

    fifo.insertStall(4, 2);

    HardwareSimulatorTester::checkFIFOInvariants(fifo);

    for (size_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(!fifo.scheduledOpInfosPerExecutor[i].empty());
        EXPECT_EQ(fifo.scheduledOpInfosPerExecutor[i].back()->getCycleEnd(), 8);
    }
}

TEST_F(MLIR_HardwareSimulator, FIFO_StressTest) {
    constexpr size_t JOB_COUNT = 10000;
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(123456);
    std::uniform_int_distribution<size_t> cycleCostDist(1, 200);

    auto fifo = HardwareSimulatorTester::makeFIFO(2);

    size_t totalCost = 0;
    for (size_t i = 0; i < JOB_COUNT; ++i) {
        const size_t cycleCost = cycleCostDist(gen);
        totalCost += cycleCost;
        const size_t requiredExecutors = cycleCostDist(gen) % 2 + 1;  // 1 or 2 executors
        const auto slot = fifo.findNextAvailableSlot(requiredExecutors);
        fifo.scheduleOp(cycleCost, slot);
    }

    EXPECT_LE(fifo.scheduledOpInfosPerExecutor[0].back()->getCycleEnd(), totalCost);
    EXPECT_LE(fifo.scheduledOpInfosPerExecutor[1].back()->getCycleEnd(), totalCost);

    HardwareSimulatorTester::checkFIFOInvariants(fifo);
}

TEST_F(MLIR_HardwareSimulator, Sim_API) {
    HardwareSimulator sim(4, {VPUIP::DmaChannelType::CMX, VPUIP::DmaChannelType::DDR});

    // There is no executor of kind DMA_NN with id 99.
    EXPECT_THROW(sim.findNextAvailableSlot(QueueType(config::ExecutorKind::DMA_NN, 99), 1), vpux::Exception);

    // No operations have been scheduled yet.
    EXPECT_THROW(sim.getScheduledOpInfo(0), vpux::Exception);
}

TEST_F(MLIR_HardwareSimulator, Sim_TypicalUsage) {
    HardwareSimulator sim(4, {VPUIP::DmaChannelType::CMX, VPUIP::DmaChannelType::DDR});

    constexpr size_t DPU_0 = 0;
    constexpr size_t DPU_1 = 1;
    constexpr size_t DPU_2 = 2;
    constexpr size_t SHAVE_NN_0 = 3;
    constexpr size_t SHAVE_NN_1 = 4;
    constexpr size_t DMA_NN_0 = 5;
    constexpr size_t DMA_NN_1 = 6;

    // op index, FIFO type, cycle cost, executor count
    const std::initializer_list<std::tuple<size_t, QueueType, size_t, size_t>> ops = {
            {DPU_0, QueueType(config::ExecutorKind::DPU), 4, 1},
            {SHAVE_NN_0, QueueType(config::ExecutorKind::SHAVE_NN), 5, 1},
            {DMA_NN_0, QueueType(config::ExecutorKind::DMA_NN), 6, 1},
            {DPU_1, QueueType(config::ExecutorKind::DPU), 3, 1},
            {SHAVE_NN_1, QueueType(config::ExecutorKind::SHAVE_NN), 9, 1},
            {DMA_NN_1, QueueType(config::ExecutorKind::DMA_NN), 8, 4}};

    for (const auto& [opIndex, queueType, cycleCost, executorCount] : ops) {
        const auto slot = sim.findNextAvailableSlot(queueType, executorCount);
        sim.scheduleOp(opIndex, cycleCost, queueType, slot);
    }

    EXPECT_EQ(sim.getScheduledOpInfo(DPU_0).cycleBegin, 0);
    EXPECT_EQ(sim.getScheduledOpInfo(DPU_1).cycleBegin, 4);

    // This should schedule on DPU executor 0 after DPU_0 but before DPU_1.
    sim.scheduleOp(DPU_2, 3, QueueType(config::ExecutorKind::DPU), {3, 0x1});

    EXPECT_EQ(sim.getScheduledOpInfo(DPU_0).cycleBegin, 0);
    EXPECT_EQ(sim.getScheduledOpInfo(DPU_1).cycleBegin, 4);
    EXPECT_EQ(sim.getScheduledOpInfo(DPU_2).cycleBegin, 3);

    sim.insertStall(4, 5);

    EXPECT_EQ(sim.getScheduledOpInfo(DPU_0).cycleBegin, 0);
    EXPECT_EQ(sim.getScheduledOpInfo(DPU_1).cycleBegin, 4 + 5);
    EXPECT_EQ(sim.getScheduledOpInfo(DPU_2).cycleBegin, 3);

    sim.insertStall(5, 10);

    EXPECT_EQ(sim.getScheduledOpInfo(DPU_0).cycleBegin, 0);
    EXPECT_EQ(sim.getScheduledOpInfo(DPU_1).cycleBegin, 4 + 5 + 10);
    EXPECT_EQ(sim.getScheduledOpInfo(DPU_2).cycleBegin, 3);

    llvm::outs() << "HardwareSimulator debug output:\n";
    llvm::outs() << formatv("{0}", sim);
}

}  // namespace
