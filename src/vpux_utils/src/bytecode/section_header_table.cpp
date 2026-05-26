//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/section_header_table.hpp"
#include "vpux/utils/bytecode/magic_number.hpp"
#include "vpux/utils/bytecode/print_utils.hpp"
#include "vpux/utils/bytecode/serialization_utils.hpp"
#include "vpux/utils/bytecode/span.hpp"
#include "vpux/utils/bytecode/version.hpp"

#include <intel_npu/utils/logger/logger.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpux;

size_t bytecode::details::FunctionSectionInfo::getBinarySize() const {
    const auto functionInfoSize = sizeof(FunctionInfo::nameIndex) + sizeof(FunctionInfo::functionTypeIndex) +
                                  sizeof(FunctionInfo::numGeneralRegisters) + sizeof(FunctionInfo::bodyOffset) +
                                  sizeof(FunctionInfo::bodySize);
    return sizeof(numFunctions) + sizeof(entrypointFunctionIndex) + functionInfos.size() * functionInfoSize;
}

void bytecode::details::FunctionSectionInfo::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, numFunctions);
    appendValueTo(buffer, entrypointFunctionIndex);
    for (const auto& functionInfo : functionInfos) {
        appendValueTo(buffer, functionInfo.nameIndex);
        appendValueTo(buffer, functionInfo.functionTypeIndex);
        appendValueTo(buffer, functionInfo.numGeneralRegisters);
        appendValueTo(buffer, functionInfo.bodyOffset);
        appendValueTo(buffer, functionInfo.bodySize);
    }
}

bool bytecode::details::FunctionSectionInfo::parseFrom(bytecode::Span<uint8_t>& buffer) {
    auto log = intel_npu::Logger::global();
    log.setName("FunctionSectionInfo::parseFrom");

    if (!parseValueFrom(buffer, numFunctions)) {
        log.error("Failed to parse numFunctions from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, entrypointFunctionIndex)) {
        log.error("Failed to parse entrypointFunctionIndex from buffer");
        return false;
    }
    for (uint64_t i = 0; i < numFunctions; ++i) {
        FunctionSectionInfo::FunctionInfo functionInfo{};
        if (!parseValueFrom(buffer, functionInfo.nameIndex)) {
            log.error("Failed to parse nameIndex from buffer, for function %d", i);
            return false;
        }
        if (!parseValueFrom(buffer, functionInfo.functionTypeIndex)) {
            log.error("Failed to parse functionTypeIndex from buffer, for function %d", i);
            return false;
        }
        if (!parseValueFrom(buffer, functionInfo.numGeneralRegisters)) {
            log.error("Failed to parse numGeneralRegisters from buffer, for function %d", i);
            return false;
        }
        if (!parseValueFrom(buffer, functionInfo.bodyOffset)) {
            log.error("Failed to parse bodyOffset from buffer, for function %d", i);
            return false;
        }
        if (!parseValueFrom(buffer, functionInfo.bodySize)) {
            log.error("Failed to parse bodySize from buffer %p %d, for function %d", buffer.begin(), buffer.size(), i);
            return false;
        }
        functionInfos.push_back(functionInfo);
    }
    return true;
}

void bytecode::details::FunctionSectionInfo::print(size_t indentLevel) const {
    bytecode::printIndent(indentLevel);
    std::cout << "Number of functions: " << numFunctions << ", entrypoint function index: " << entrypointFunctionIndex
              << std::endl;
    for (const auto& functionInfo : functionInfos) {
        bytecode::printIndent(indentLevel + 1);
        // TODO: Replace name index and type index with actual entries from string / type sections
        std::cout << "Name index: " << functionInfo.nameIndex
                  << ", function type index: " << functionInfo.functionTypeIndex
                  << ", num general registers: " << functionInfo.numGeneralRegisters
                  << ", body offset: " << functionInfo.bodyOffset << ", body size: " << functionInfo.bodySize
                  << std::endl;
    }
}

size_t bytecode::details::DataSectionInfo::getBinarySize() const {
    const auto dataInfoSize = sizeof(DataInfo::offset) + sizeof(DataInfo::size);
    return sizeof(numData) + dataInfos.size() * dataInfoSize;
}

void bytecode::details::DataSectionInfo::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, numData);
    for (const auto& dataInfo : dataInfos) {
        appendValueTo(buffer, dataInfo.offset);
        appendValueTo(buffer, dataInfo.size);
    }
}

bool bytecode::details::DataSectionInfo::parseFrom(bytecode::Span<uint8_t>& buffer) {
    auto log = intel_npu::Logger::global();
    log.setName("DataSectionInfo::parseFrom");

    if (!parseValueFrom(buffer, numData)) {
        log.error("Failed to parse numData from buffer");
        return false;
    }
    for (uint64_t i = 0; i < numData; ++i) {
        DataSectionInfo::DataInfo dataInfo{};
        if (!parseValueFrom(buffer, dataInfo.offset)) {
            log.error("Failed to parse offset from buffer, for data %d", i);
            return false;
        }
        if (!parseValueFrom(buffer, dataInfo.size)) {
            log.error("Failed to parse size from buffer, for data %d", i);
            return false;
        }
        dataInfos.push_back(dataInfo);
    }
    return true;
}

void bytecode::details::DataSectionInfo::print(size_t indentLevel) const {
    bytecode::printIndent(indentLevel);
    std::cout << "Number of entries: " << numData << std::endl;
    for (uint64_t i = 0; i < dataInfos.size(); ++i) {
        const auto& dataInfo = dataInfos[i];
        bytecode::printIndent(indentLevel + 1);
        std::cout << "Entry " << i << " offset: " << dataInfo.offset << ", size: " << dataInfo.size << std::endl;
    }
}

std::string bytecode::getSectionTypeString(bytecode::SectionType type) {
    switch (type) {
    case SectionType::FuncSection:
        return "Function";
    case SectionType::ConstantSection:
        return "Constant";
    case SectionType::StringSection:
        return "String";
    case SectionType::KernelSection:
        return "Kernel";
    case SectionType::TypeSection:
        return "Type";
    default:
        return "Unknown";
    }
}

size_t bytecode::SectionHeader::getBinarySize() const {
    return sizeof(type) + sizeof(nameIndex) + sizeof(offset) + sizeof(size) + info->getBinarySize();
}

void bytecode::SectionHeader::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, static_cast<uint8_t>(type));
    appendValueTo(buffer, nameIndex);
    appendValueTo(buffer, offset);
    appendValueTo(buffer, size);
    info->appendTo(buffer);
}

bool bytecode::SectionHeader::parseFrom(bytecode::Span<uint8_t>& buffer) {
    auto log = intel_npu::Logger::global();
    log.setName("SectionHeader::parseFrom");

    if (!parseValueFrom(buffer, type)) {
        log.error("Failed to parse type from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, nameIndex)) {
        log.error("Failed to parse nameIndex from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, offset)) {
        log.error("Failed to parse offset from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, size)) {
        log.error("Failed to parse size from buffer");
        return false;
    }
    if (type == SectionType::FuncSection) {
        auto functionSectionInfo = details::FunctionSectionInfo{};
        if (!functionSectionInfo.parseFrom(buffer)) {
            log.error("Failed to parse function section info from buffer");
            return false;
        }
        info = std::make_unique<details::FunctionSectionInfo>(functionSectionInfo);
    } else if (type == SectionType::ConstantSection || type == SectionType::KernelSection ||
               type == SectionType::StringSection || type == SectionType::TypeSection) {
        auto dataSectionInfo = details::DataSectionInfo{};
        if (!dataSectionInfo.parseFrom(buffer)) {
            log.error("Failed to parse data section info from buffer");
            return false;
        }
        info = std::make_unique<details::DataSectionInfo>(dataSectionInfo);
    } else {
        log.error("Unknown section type {0}", static_cast<uint8_t>(type));
        return false;
    }
    return true;
}

void bytecode::SectionHeader::print(size_t indentLevel) const {
    bytecode::printIndent(indentLevel);
    std::cout << "Section type: " << getSectionTypeString(type) << ", name index: " << nameIndex
              << ", offset: " << offset << ", size: " << size << std::endl;
    info->print(indentLevel + 1);
}

std::vector<bytecode::SectionHeader>& bytecode::SectionHeaderTable::getSectionHeaders() {
    return _sectionHeaders;
}

void bytecode::SectionHeaderTable::addSectionHeader(SectionHeader header) {
    _sectionHeaders.push_back(std::move(header));
    ++_numSections;
}

size_t bytecode::SectionHeaderTable::getBinarySize() const {
    size_t size = sizeof(_numSections);
    for (const auto& header : _sectionHeaders) {
        size += header.getBinarySize();
    }
    return size;
}

void bytecode::SectionHeaderTable::computeOffsets() {
    // Section payloads start immediately after the file header, which consists of magic number, version, and
    // the section header table itself
    uint64_t currentOffset = MagicNumber::getBinarySize() + Version::getBinarySize() + getBinarySize();
    // Assign offsets in a deterministic order so that the file layout is predictable: functions first, then constants,
    // strings, and finally types
    const auto updateOffsets = [&](SectionType sectionType) {
        for (auto& header : _sectionHeaders) {
            if (header.type == sectionType) {
                header.offset = currentOffset;
                currentOffset += header.size;
            }
        }
    };
    updateOffsets(SectionType::FuncSection);
    updateOffsets(SectionType::ConstantSection);
    updateOffsets(SectionType::KernelSection);
    updateOffsets(SectionType::StringSection);
    updateOffsets(SectionType::TypeSection);
}

void bytecode::SectionHeaderTable::appendTo(std::vector<uint8_t>& buffer) const {
    // Serialize headers grouped by section type to match the offset assignment order used in computeOffsets()
    const auto appendSection = [&](SectionType sectionType) {
        for (const auto& header : _sectionHeaders) {
            if (header.type == sectionType) {
                header.appendTo(buffer);
            }
        }
    };
    appendValueTo(buffer, _numSections);
    appendSection(SectionType::FuncSection);
    appendSection(SectionType::ConstantSection);
    appendSection(SectionType::KernelSection);
    appendSection(SectionType::StringSection);
    appendSection(SectionType::TypeSection);
}

bool bytecode::SectionHeaderTable::parseFrom(bytecode::Span<uint8_t>& buffer) {
    auto log = intel_npu::Logger::global();
    log.setName("SectionHeaderTable::parseFrom");

    uint64_t fileNumSections = 0;
    if (!parseValueFrom(buffer, fileNumSections)) {
        log.error("Failed to parse numSections from buffer");
        return false;
    }
    for (uint64_t i = 0; i < fileNumSections; ++i) {
        SectionHeader header{};
        if (!header.parseFrom(buffer)) {
            log.error("Failed to parse section header %d from buffer", i);
            return false;
        }
        addSectionHeader(std::move(header));
    }
    return true;
}

void bytecode::SectionHeaderTable::print(size_t indentLevel) const {
    bytecode::printIndent(indentLevel);
    std::cout << "Number of sections: " << _numSections << std::endl;
    for (const auto& header : _sectionHeaders) {
        header.print(indentLevel + 1);
    }
}
