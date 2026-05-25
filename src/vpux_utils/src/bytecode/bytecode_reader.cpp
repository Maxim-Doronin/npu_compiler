//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/bytecode_reader.hpp"
#include "vpux/utils/bytecode/magic_number.hpp"
#include "vpux/utils/bytecode/print_utils.hpp"
#include "vpux/utils/bytecode/section_header_table.hpp"
#include "vpux/utils/bytecode/span.hpp"
#include "vpux/utils/bytecode/version.hpp"

#include <intel_npu/utils/logger/logger.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace vpux;

bytecode::SectionHeaderTable& bytecode::BytecodeReader::getSectionHeaderTable() {
    return _sectionHeaderTable;
}

const std::vector<std::vector<uint8_t>>& bytecode::BytecodeReader::getSections() const {
    return _sections;
}

bool bytecode::BytecodeReader::parseFileHeader() {
    auto log = intel_npu::Logger::global();
    log.setName("BytecodeReader::parseFileHeader");

    // Wrap the raw bytecode in a Span that tracks the current read position.
    // Each parseFrom() call advances the span past the consumed bytes.
    auto bytecodeBuffer = bytecode::Span<uint8_t>{_bytecode.data(), _bytecode.size()};

    if (!_magicNumber.parseFrom(bytecodeBuffer)) {
        log.error("Failed to parse magic number");
        return false;
    }

    if (!_version.parseFrom(bytecodeBuffer)) {
        log.error("Failed to parse version");
        return false;
    }

    if (!_sectionHeaderTable.parseFrom(bytecodeBuffer)) {
        log.error("Failed to parse section header table");
        return false;
    }

    return true;
}

bool bytecode::BytecodeReader::parseSections() {
    auto log = intel_npu::Logger::global();
    log.setName("BytecodeReader::parseSections");

    auto bytecodeBuffer = bytecode::Span<uint8_t>{_bytecode.data(), _bytecode.size()};

    for (const auto& header : _sectionHeaderTable.getSectionHeaders()) {
        const auto sectionType = header.type;
        const auto offset = header.offset;
        const auto size = header.size;
        if (offset + size > bytecodeBuffer.size()) {
            log.error("Section of type %d has invalid offset and size that exceed bytecode buffer",
                      static_cast<uint64_t>(sectionType));
            return false;
        }
        auto sectionBuffer = bytecodeBuffer.subspan(offset, size);
        _sections.emplace_back(sectionBuffer.begin(), sectionBuffer.end());
    }

    return true;
}

bool bytecode::BytecodeReader::parseFile() {
    auto log = intel_npu::Logger::global();
    log.setName("BytecodeReader::parseFile");
    if (!parseFileHeader()) {
        log.error("Failed to parse file header");
        return false;
    }
    if (!parseSections()) {
        log.error("Failed to parse sections");
        return false;
    }
    return true;
}

void bytecode::BytecodeReader::printFileHeader(size_t indentLevel) {
    bytecode::printIndent(indentLevel);
    std::cout << "Magic Number: ";
    _magicNumber.print();
    std::cout << std::endl;

    bytecode::printIndent(indentLevel);
    std::cout << "Version: ";
    _version.print();
    std::cout << std::endl;

    bytecode::printIndent(indentLevel);
    std::cout << "Section Header Table:" << std::endl;
    _sectionHeaderTable.print(indentLevel + 1);
}

void bytecode::BytecodeReader::printFile(bool printFull, size_t indentLevel) {
    printFileHeader(indentLevel);

    size_t functionSectionIdx = 0;
    size_t constantSectionIdx = 0;
    size_t kernelSectionIdx = 0;
    size_t stringSectionIdx = 0;
    size_t typeSectionIdx = 0;

    auto& sectionHeaders = _sectionHeaderTable.getSectionHeaders();
    for (size_t i = 0; i < sectionHeaders.size(); ++i) {
        if (i >= _sections.size()) {
            std::cerr << "Error: Could not find associated section with section header " << i << std::endl;
            continue;
        }
        auto sectionContent = bytecode::Span<uint8_t>(_sections[i].data(), _sections[i].size());

        const auto& header = sectionHeaders[i];
        if (header.type == SectionType::FuncSection) {
            printFunctionSection(header, sectionContent, functionSectionIdx++, indentLevel + 1);
        } else if (header.type == SectionType::ConstantSection) {
            printDataSection(header, sectionContent, constantSectionIdx++, header.type, printFull, indentLevel + 1);
        } else if (header.type == SectionType::StringSection) {
            printDataSection(header, sectionContent, stringSectionIdx++, header.type, printFull, indentLevel + 1);
        } else if (header.type == SectionType::KernelSection) {
            printDataSection(header, sectionContent, kernelSectionIdx++, header.type, printFull, indentLevel + 1);
        } else if (header.type == SectionType::TypeSection) {
            printDataSection(header, sectionContent, typeSectionIdx++, header.type, printFull, indentLevel + 1);
        } else {
            std::cerr << "Error: Unsupported section type for printing: " << static_cast<uint64_t>(header.type)
                      << std::endl;
        }
    }
}
