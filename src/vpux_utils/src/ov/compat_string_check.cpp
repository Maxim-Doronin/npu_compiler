//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/ov/compat_string_check.hpp"
#include "vpux/utils/ov/compat_string_parser.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace {

uint64_t parseInt(const std::string& str) {
    size_t pos = 0;
    uint64_t value = std::stoull(str, &pos);
    if (pos != str.size()) {
        throw std::runtime_error("Invalid integer: " + str);
    }
    return value;
}

}  // namespace

namespace vpux::compat {

BlobRequirements parseCompatibilityString(std::string_view compatibilityString) {
    parser::Parser parser(compatibilityString);

    // Validate that only known attributes are present
    std::array legalAttributes = {"npu", "t", "compiler", "elf", "mi"};
    for (const auto& [name, value] : parser.getAttributes()) {
        if (std::find(legalAttributes.begin(), legalAttributes.end(), name) == legalAttributes.end()) {
            throw std::runtime_error("Illegal attribute in compatibility string: " + name);
        }
    }

    BlobRequirements reqs;
    reqs.platformId = parseInt(parser.getAttribute("npu"));
    reqs.numTiles = parseInt(parser.getAttribute("t"));
    return reqs;
}

}  // namespace vpux::compat
