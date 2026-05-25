//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/bytecode/magic_number.hpp"
#include "vpux/utils/bytecode/section_header_table.hpp"
#include "vpux/utils/bytecode/version.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vpux::bytecode {

// Deserializes a bytecode binary into its constituent parts: file header and section payloads
class BytecodeReader {
    std::vector<uint8_t> _bytecode;  // Raw bytecode binary held for the lifetime of the reader

    MagicNumber _magicNumber{};                   // File format identifier parsed from the bytecode header
    Version _version{};                           // Bytecode format version (major.minor.patch)
    SectionHeaderTable _sectionHeaderTable;       // Table of contents describing all sections in the file
    std::vector<std::vector<uint8_t>> _sections;  // Raw payload bytes for each section, indexed in header order

    // Parses the file header fields (magic number, version, section header table) from the beginning of the bytecode
    // buffer. Returns false on malformed input.
    bool parseFileHeader();

    // Extracts raw section payloads using offsets and sizes from the previously
    // parsed section header table. Must be called after parseFileHeader().
    // Returns false if any section exceeds the bytecode buffer bounds.
    bool parseSections();

public:
    // Constructs a reader that takes ownership of the raw bytecode bytes
    BytecodeReader(std::vector<uint8_t> bytecode): _bytecode(std::move(bytecode)), _sectionHeaderTable() {
    }

    // Returns a reference to the parsed section header table. Valid only after a successful parseFile() call.
    SectionHeaderTable& getSectionHeaderTable();

    // Returns the extracted section payloads as a list of byte vectors,
    // ordered to match the section headers. Valid only after a successful parseFile() call.
    const std::vector<std::vector<uint8_t>>& getSections() const;

    // Parses the entire bytecode file: header followed by section payloads.
    // Returns false and logs an error if any stage of parsing fails.
    bool parseFile();

    // Prints the parsed file header to stdout
    void printFileHeader(size_t indentLevel = 0);

    /// Prints the entire parsed file to stdout
    /// @param printFull If true, also prints the content of binary sections such constants or kernels
    /// @param indentLevel The indentation level for pretty-printing nested structures
    void printFile(bool printFull = true, size_t indentLevel = 0);
};

}  // namespace vpux::bytecode
