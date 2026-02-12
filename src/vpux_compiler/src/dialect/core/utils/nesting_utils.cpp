//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/utils/nesting_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

static constexpr auto MODE_DEFAULT = "default";
static constexpr auto MODE_ENTRY_POINT = "entry-point";

Core::NestingMode Core::parseNestingMode(std::string& mode) {
    for (char& c : mode) {
        c = static_cast<char>(std::tolower(c));
    }

    if (mode == MODE_DEFAULT) {
        return Core::NestingMode::Default;
    } else if (mode == MODE_ENTRY_POINT) {
        return Core::NestingMode::EntryPoint;
    }

    VPUX_THROW("Unknown value for nesting mode option: {0}", mode);
}
