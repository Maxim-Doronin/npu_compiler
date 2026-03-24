//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {

/** @brief Class that manages SHAVE cost model utilities
 *  Stores the list of supported SHAVE operations retrieved from VPUCostModel.
 */
class CostModelShaveUtil {
public:
    // Check if a kernel is supported
    bool isSwKernelOpSupported(const std::string& swKernelName) const {
        return std::find(_supportedOperations.begin(), _supportedOperations.end(), swKernelName) !=
               _supportedOperations.end();
    }

    // Check if Shave2 API is used
    bool isShave2ApiUsed() const {
        return _isShave2ApiUsedInVPUNN;
    }

    CostModelShaveUtil(bool isShave2ApiUsed, const std::vector<std::string>& supportedOperations)
            : _isShave2ApiUsedInVPUNN(isShave2ApiUsed), _supportedOperations(supportedOperations) {
    }

private:
    bool _isShave2ApiUsedInVPUNN;
    std::vector<std::string> _supportedOperations;
};
}  // namespace VPU
}  // namespace vpux
