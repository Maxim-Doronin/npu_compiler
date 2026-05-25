//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace vpux::bytecode {

// Instruction opcodes recognized by the bytecode interpreter
enum class OpCode : uint16_t {
    ADD_I32 = 0x01,
    ADD_I64 = 0x02,
    RET = 0x03,
    SET = 0x04,
    SET_IMM = 0x05,
    ASSERT = 0x06,
    MUL_I32 = 0x07,
    MUL_I64 = 0x08,
    MIN_I32 = 0x09,
    MIN_I64 = 0x0A,
    MAX_I32 = 0x0B,
    MAX_I64 = 0x0C,
};

// The byte size of every instruction
inline const std::unordered_map<OpCode, size_t> INSTRUCTION_SIZES = {
        {OpCode::ADD_I32, sizeof(uint16_t) + 3 * sizeof(int16_t)},  // opcode + 3 operands
        {OpCode::ADD_I64, sizeof(uint16_t) + 3 * sizeof(int16_t)},  // opcode + 3 operands
        {OpCode::RET, sizeof(uint16_t)},                            // opcode
        {OpCode::SET, sizeof(uint16_t) + 2 * sizeof(int16_t)},      // opcode + 2 operands
        {OpCode::SET_IMM,
         sizeof(uint16_t) + sizeof(int16_t) + sizeof(int64_t)},     // opcode + 1 16-bit operand + 1 64-bit operand
        {OpCode::ASSERT, sizeof(uint16_t) + 2 * sizeof(int16_t)},   // opcode + 2 operands
        {OpCode::MUL_I32, sizeof(uint16_t) + 3 * sizeof(int16_t)},  // opcode + 3 operands
        {OpCode::MUL_I64, sizeof(uint16_t) + 3 * sizeof(int16_t)},  // opcode + 3 operands
        {OpCode::MIN_I32, sizeof(uint16_t) + 3 * sizeof(int16_t)},  // opcode + 3 operands
        {OpCode::MIN_I64, sizeof(uint16_t) + 3 * sizeof(int16_t)},  // opcode + 3 operands
        {OpCode::MAX_I32, sizeof(uint16_t) + 3 * sizeof(int16_t)},  // opcode + 3 operands
        {OpCode::MAX_I64, sizeof(uint16_t) + 3 * sizeof(int16_t)},  // opcode + 3 operands
};

// Get the opcode of the instruction whose binary representation starts at 'instructionBegin'
inline uint16_t getOpcode(const uint8_t* instructionBegin) {
    uint16_t value = 0;
    std::memcpy(&value, instructionBegin, sizeof(value));
    return value;
}

// The value of the i-th operand for the instruction whose binary representation starts at 'begin'.
// This util is intended to be used only for instructions that have operands represented as 16-bit signed integers
inline int16_t getOperand(const uint8_t* instructionBegin, size_t operandIndex) {
    int16_t value = 0;
    std::memcpy(&value, instructionBegin + sizeof(OpCode) + operandIndex * sizeof(int16_t), sizeof(value));
    return value;
}

inline int64_t get64BitImm(const uint8_t* instructionBegin, size_t operandIndex) {
    int64_t value = 0;
    std::memcpy(&value, instructionBegin + sizeof(OpCode) + operandIndex * sizeof(int16_t), sizeof(value));
    return value;
}

}  // namespace vpux::bytecode
