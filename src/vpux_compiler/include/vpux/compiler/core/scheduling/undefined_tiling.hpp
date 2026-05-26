//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/scheduling/temporal_tiling_interface.hpp"

namespace vpux {

// Fallback tiling scenario used when no specific strategy (isolated, prefetch, or pipelining)
// has been assigned to a loop region. Provides a general-purpose memory allocation and
// schedule generation that works for any buffer pattern, trading optimal performance
// for universal applicability.
// E-197030: Introduce specific tiling scenarios (isolated, prefetch, pipelining).
class UndefinedTiling : public ITemporalTilingScenario {
public:
    UndefinedTiling();
    llvm::StringRef getName() const override;
    // Generates a schedule for the given loop region and memory size using a simple allocation strategy:
    //   1. Analyze buffer usage across iterations to identify shared buffers and dependencies.
    //   2. Reserve (not allocate) shared buffers from top to down in CMX.
    //   3. Allocate per-iteration buffers in a ping-pong manner, alternating between two slots to minimize reloads.
    LoopScheduleResult getScheduleStrategy(const ComputeRegion& loopRegion,
                                           vpux::AddressType memorySize) const override;

private:
    Logger _log;
};

}  // namespace vpux
