//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/version.hpp"
#include "vpux/utils/bytecode/print_utils.hpp"
#include "vpux/utils/bytecode/serialization_utils.hpp"
#include "vpux/utils/bytecode/span.hpp"

#include <intel_npu/utils/logger/logger.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

using namespace vpux;

size_t bytecode::Version::getBinarySize() {
    using VersionType = decltype(std::declval<bytecode::Version>().getFullVersion());
    return sizeof(VersionType);
}

void bytecode::Version::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, _version);
}

bool bytecode::Version::parseFrom(bytecode::Span<uint8_t>& buffer) {
    auto log = intel_npu::Logger::global();
    log.setName("Version::parseFrom");
    if (!parseValueFrom(buffer, _version)) {
        log.error("Failed to parse version from buffer");
        return false;
    }
    return true;
}

void bytecode::Version::print(size_t indentLevel) const {
    bytecode::printIndent(indentLevel);
    std::cout << getMajor() << "." << getMinor() << "." << getPatch();
}
