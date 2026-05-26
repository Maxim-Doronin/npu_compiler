//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/magic_number.hpp"
#include "vpux/utils/bytecode/print_utils.hpp"
#include "vpux/utils/bytecode/serialization_utils.hpp"
#include "vpux/utils/bytecode/span.hpp"

#include <intel_npu/utils/logger/logger.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace vpux;

const std::array<uint8_t, bytecode::MAGIC_NUMBER_SIZE>& bytecode::MagicNumber::value() const {
    return _value;
}

size_t bytecode::MagicNumber::getBinarySize() {
    return MAGIC_NUMBER_SIZE * sizeof(uint8_t);
}

void bytecode::MagicNumber::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, _value);
}

bool bytecode::MagicNumber::parseFrom(bytecode::Span<uint8_t>& buffer) {
    auto log = intel_npu::Logger::global();
    log.setName("MagicNumber::parseFrom");
    if (!parseValueFrom(buffer, _value)) {
        log.error("Failed to parse magic number from buffer");
        return false;
    }
    if (!std::equal(MAGIC_NUMBER.begin(), MAGIC_NUMBER.end(), _value.begin(), _value.end())) {
        log.error("Magic number does not match expected value");
        return false;
    }
    return true;
}

void bytecode::MagicNumber::print(size_t indentLevel) const {
    bytecode::printIndent(indentLevel);
    for (const auto& byte : _value) {
        std::cout << bytecode::toHex(byte) << " ";
    }
}
