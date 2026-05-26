//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/// @file loop_schedule_utils.cpp
/// @brief Predefined loop schedule generation for tiled compute regions.
///
/// Transforms ComputeRegion objects (produced by getComputeRegionsFromAsyncExec)
/// into predefined schedules that FeasibleMemoryScheduler replays during loop
/// region scheduling. Each recognized LoopType is mapped to a temporal tiling
/// strategy that determines buffer allocation order and ping-pong placement.
///
/// Usage: called in FeasibleAllocationPass after ComputeRegion extraction
/// before FeasibleMemoryScheduler construction which applies predefined schedules.

#include "vpux/compiler/dialect/VPUIP/utils/loop_schedule_utils.hpp"

#include "vpux/compiler/core/scheduling/undefined_tiling.hpp"

using namespace vpux;

namespace {

/// Map from LoopType to temporal tiling scenario name identifier
std::map<LoopType, std::string> getLoopTypeToScenarioMap() {
    return {{LoopType::Tiling, "UNDEFINED_TILING"}};
}

/// Create temporal tiling scenario instances keyed by scenario name
std::map<std::string, std::unique_ptr<ITemporalTilingScenario>> createTemporalTilingScenarios() {
    std::map<std::string, std::unique_ptr<ITemporalTilingScenario>> scenarios;
    scenarios["UNDEFINED_TILING"] = std::make_unique<UndefinedTiling>();
    return scenarios;
}

}  // namespace

ComputeRegionsSchedule vpux::VPUIP::generateLoopSchedules(const ComputeRegionVec& loopRegions,
                                                          vpux::AddressType memorySize, Logger log) {
    ComputeRegionsSchedule computeRegionsSchedule;

    const auto loopTypeToScenarioMap = getLoopTypeToScenarioMap();
    auto temporalTilingScenarios = createTemporalTilingScenarios();

    for (size_t idx = 0; idx < loopRegions.size(); ++idx) {
        const auto& computeRegion = loopRegions[idx];
        if (computeRegion.getLoopType() != LoopType::None) {
            log.trace("{0}", computeRegion);
        }
        if (computeRegion.getLoopType() == LoopType::None) {
            continue;
        }

        const auto scenarioIt = loopTypeToScenarioMap.find(computeRegion.getLoopType());
        if (scenarioIt == loopTypeToScenarioMap.end()) {
            continue;
        }

        const auto& scenarioName = scenarioIt->second;
        auto scenarioImplIt = temporalTilingScenarios.find(scenarioName);
        VPUX_THROW_UNLESS(scenarioImplIt != temporalTilingScenarios.end(), "Temporal tiling scenario '{0}' not found",
                          scenarioName);
        auto result = scenarioImplIt->second->getScheduleStrategy(computeRegion, memorySize);
        if (result.empty()) {
            continue;
        }

        // Categorize operations from loop bodies into scheduling sets
        for (const auto& loop : computeRegion.schedulingLoop->loopBodies) {
            for (const auto& alloc : loop) {
                if (alloc.allocationType == AllocationType::DATA_IN) {
                    // DATA_IN operations (input DMAs) can be prefetched normally
                    computeRegionsSchedule.loopPrefetchInd.insert(alloc.opIdx);
                    continue;
                }
                // All other operations (COMPUTE, DATA_OUT, etc.) require loop scheduling
                computeRegionsSchedule.loopRegionInd.insert(alloc.opIdx);
            }
        }

        computeRegionsSchedule.scheduleResults[idx] = std::move(result);
    }

    return computeRegionsSchedule;
}
