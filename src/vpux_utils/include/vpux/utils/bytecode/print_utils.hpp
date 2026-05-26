//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/bytecode/section_header_table.hpp"
#include "vpux/utils/bytecode/span.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace vpux::bytecode {

inline void printIndent(size_t indentLevel) {
    constexpr auto numSpaces = 2;
    for (size_t i = 0; i < indentLevel * numSpaces; ++i) {
        std::cout << " ";
    }
}

std::string toHex(uint8_t byte);

void printFunctionSection(const bytecode::SectionHeader& sectionHeader, bytecode::Span<uint8_t> sectionContent,
                          size_t sectionIdx, size_t indentLevel);
void printDataSection(const bytecode::SectionHeader& sectionHeader, bytecode::Span<uint8_t> sectionContent,
                      size_t sectionIdx, bytecode::SectionType sectionType, bool printFull, size_t indentLevel);

}  // namespace vpux::bytecode
