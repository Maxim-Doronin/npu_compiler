//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"

namespace vpux {

// Interface for temporal tiling scenarios used in Tiling and loop scheduling.
//
// Each scenario represents a CMX memory allocation strategy with different performance
// and memory trade-offs:
//   - Isolated tiling:   no overlap between iterations.
//   - Prefetch tiling:   overlaps data transfers with computation.
//   - Pipelining tiling: full prefetching and double-buffering.
//   - Undefined tiling:  fallback when no specific scenario is assigned.
//
// The interface establishes a contract between tiling and scheduling:
//   - Tiling selects the appropriate scenario based on CMX occupancy constraints.
//   - Scheduling queries the scenario for its optimal loop allocation strategy
//     via getScheduleStrategy(), producing deterministic and optimized schedules.
//
// E-197030: Current implementation only focuses on loop scheduling, the usage for tiling is planned in future
// iterations. Current implementation only focuses on the UndefinedTiling scenario, which handles all  cases. Other
// scenarios will be implemented and integrated in future iterations.

class ITemporalTilingScenario {
public:
    ITemporalTilingScenario() = default;
    virtual ~ITemporalTilingScenario() = default;
    ITemporalTilingScenario(const ITemporalTilingScenario&) = delete;
    ITemporalTilingScenario& operator=(const ITemporalTilingScenario&) = delete;
    ITemporalTilingScenario(ITemporalTilingScenario&&) = delete;
    ITemporalTilingScenario& operator=(ITemporalTilingScenario&&) = delete;

    virtual llvm::StringRef getName() const = 0;
    virtual LoopScheduleResult getScheduleStrategy(const ComputeRegion& loopRegion,
                                                   vpux::AddressType memorySize) const = 0;
};

}  // namespace vpux
