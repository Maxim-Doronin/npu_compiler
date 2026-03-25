//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"

namespace vpux {

// Test helper class for FeasibleMemoryScheduler to access private members
class FeasibleMemorySchedulerTest {
public:
    explicit FeasibleMemorySchedulerTest(const FeasibleMemoryScheduler& scheduler): _scheduler(scheduler) {
    }

    size_t getLoopRegionSize() const {
        return _scheduler._loopRegions.size();
    }

    size_t getScheduledLoopRegionSize() const {
        return _scheduler._scheduledLoopRegionInd.size();
    }

    // In Loop region, the input buffers are allocated to the same addresses
    bool verifyTilingLoopInputAddress() const;

private:
    const FeasibleMemoryScheduler& _scheduler;
};

}  // namespace vpux
