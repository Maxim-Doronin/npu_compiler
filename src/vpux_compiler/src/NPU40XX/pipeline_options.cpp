//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

namespace vpux {

void setupPWLMCompilationParams(int optimizationLevel, DefaultHWOptions40XX& compilationOptions, bool useWlm) {
    if (!useWlm) {
        compilationOptions.workloadManagementEnable = false;
        return;
    }
    bool isworkloadManagementEnableSet = compilationOptions.workloadManagementEnable.hasValue() ? true : false;

    if (!isworkloadManagementEnableSet) {
        std::optional<int> originalValueBarrierCountThreshold = std::nullopt;
        std::optional<WorkloadManagementMode> originalValueWorkloadManagementMode = std::nullopt;

        if (compilationOptions.workloadManagementMode.hasValue()) {
            originalValueWorkloadManagementMode = compilationOptions.workloadManagementMode;
        }

        if (compilationOptions.workloadManagementBarrierCountThreshold.hasValue()) {
            originalValueBarrierCountThreshold = compilationOptions.workloadManagementBarrierCountThreshold;
        }

        switch (optimizationLevel) {
        case 0:
            compilationOptions.workloadManagementEnable = false;
            break;
        case 1:
            compilationOptions.workloadManagementEnable = true;
            break;
        case 2: {
            compilationOptions.workloadManagementEnable = true;
            compilationOptions.workloadManagementBarrierCountThreshold = std::numeric_limits<int>::max();
            break;
        }
        case 3: {
            compilationOptions.workloadManagementEnable = true;
            compilationOptions.workloadManagementBarrierCountThreshold = std::numeric_limits<int>::max();
            compilationOptions.workloadManagementMode = WorkloadManagementMode::PWLM_V1_BARRIER_FIFO;
            break;
        }
        default:
            VPUX_THROW("Unexpected optimization-level. Actual value = {0}\n"
                       "Possible values: 0 - optimization for compilation time, "
                       "1 - optimization for execution time (default), 2 - high optimization for execution time, 3 - "
                       "optimization for maximaze HW utilization, may affect compilation time and memory footprint",
                       optimizationLevel);
            break;
        }

        if (originalValueWorkloadManagementMode.has_value()) {
            compilationOptions.workloadManagementMode = originalValueWorkloadManagementMode.value();
        }

        if (originalValueBarrierCountThreshold.has_value()) {
            compilationOptions.workloadManagementBarrierCountThreshold = originalValueBarrierCountThreshold.value();
        }
    }
}

}  // namespace vpux
