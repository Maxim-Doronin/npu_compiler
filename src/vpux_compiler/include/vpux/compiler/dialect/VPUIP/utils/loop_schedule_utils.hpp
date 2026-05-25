//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux::VPUIP {

/// Generate predefined loop schedules for all compute regions.
/// Runs temporal tiling strategy on each region that has a recognized loop type
/// and collects the results into a ComputeRegionsSchedule.
/// This function must be called before constructing FeasibleMemoryScheduler.
///
/// @param loopRegions  Compute regions extracted from async operations
/// @param memorySize   Total available CMX memory size
/// @param log          Logger instance
/// @return ComputeRegionsSchedule containing predefined schedules and operation index sets
ComputeRegionsSchedule generateLoopSchedules(const ComputeRegionVec& loopRegions, vpux::AddressType memorySize,
                                             Logger log);

}  // namespace vpux::VPUIP
