//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/bytecode/section_header_table.hpp"
#include "vpux/utils/core/array_ref.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpux::bytecode {

// Serializes an MLIR ModuleOp into a flat binary bytecode format.
// The writer builds an in-memory buffer containing the file header and serialized section payloads, then writes the
// complete buffer to an output stream.
class BytecodeWriter {
    mlir::ModuleOp _moduleOp;
    std::vector<uint8_t> _bytecodeBuffer;
    SectionHeaderTable _sectionHeaderTable;

    void prepareSectionHeaderTable();

public:
    // Construct a BytecodeWriter for the given module.
    // The section header table is prepared eagerly during construction.
    explicit BytecodeWriter(mlir::ModuleOp moduleOp);

    // Append the file header (magic number, version, section header table) to the internal bytecode buffer
    void appendFileHeader();

    // Serialize every section body (functions, constants, strings, types)
    // and append the resulting bytes to the internal bytecode buffer
    void appendSections();

    /// Encode a single instruction and append it to the bytecode buffer.
    /// @param opcode         base opcode value
    /// @param addressingMode addressing-mode bits
    /// @param operands       variable-length operand list
    void appendInstruction(uint16_t opcode, uint16_t addressingMode, ArrayRef<int16_t> operands);

    /// Encode a single instruction and append it to the bytecode buffer.
    /// @param opcode         base opcode value
    /// @param addressingMode addressing-mode bits
    /// @param binaryOperands binary representation of the operands to append directly to the instruction (used for
    /// instructions whose operands are not only 16-bit integers)
    void appendInstruction(uint16_t opcode, uint16_t addressingMode, ArrayRef<uint8_t> binaryOperands);

    /// Append raw binary data to the bytecode buffer
    /// @param data pointer to the beginning of the data
    /// @param size size of the data in bytes
    void appendRawData(const uint8_t* data, size_t size);

    // Flush the accumulated bytecode buffer to the provided output stream
    void writeTo(llvm::raw_ostream& os);
};

}  // namespace vpux::bytecode
