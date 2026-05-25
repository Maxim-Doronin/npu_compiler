//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string_view>

namespace vpux::compat {

struct BlobRequirements {
    uint64_t platformId;
    uint64_t numTiles;
};

BlobRequirements parseCompatibilityString(std::string_view compatibilityString);

}  // namespace vpux::compat
