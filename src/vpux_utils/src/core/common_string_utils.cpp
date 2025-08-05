//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/core/common_string_utils.hpp"

#include <algorithm>
#include <cstdarg>
#include <stdexcept>

namespace vpux {
void splitRangeAndApply(std::string_view::const_iterator begin, std::string_view::const_iterator end, char delim,
                        std::function<void(std::string_view)> callback) {
    auto curBegin = begin;
    auto curEnd = begin;
    while (curEnd != end) {
        while (curEnd != end && *curEnd != delim) {
            ++curEnd;
        }

        callback(std::string_view(&(*curBegin), static_cast<size_t>(curEnd - curBegin)));

        if (curEnd != end) {
            ++curEnd;
            curBegin = curEnd;
        }
    }
}
}  // namespace vpux
