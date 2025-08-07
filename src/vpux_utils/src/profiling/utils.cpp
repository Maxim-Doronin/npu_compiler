//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/profiling/utils.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/profiling/common.hpp"

namespace vpux::profiling {
std::string formatDuration(uint64_t timeNs) {
    uint64_t ms = timeNs / 1000000;
    uint64_t us = (timeNs % 1000000) / 1000;
    uint64_t ns = timeNs % 1000;
    std::string timeString;
    if (ms != 0) {
        timeString += printToString("{0}ms ", ms);
    }
    if (us != 0) {
        timeString += printToString("{0}us ", us);
    }
    timeString += printToString("{0}ns", ns);
    return timeString;
}
}  // namespace vpux::profiling
