//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//
// Various string manipulation utility functions.
//

#pragma once

#include <functional>
#include <string_view>

namespace vpux {
void splitRangeAndApply(std::string_view::const_iterator begin, std::string_view::const_iterator end, char delim,
                        std::function<void(std::string_view)> callback);
}  // namespace vpux
