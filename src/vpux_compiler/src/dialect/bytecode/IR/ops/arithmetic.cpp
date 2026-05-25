//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/bytecode/instructions.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <cstdint>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/arithmetic.cpp.inc>

void bytecode::AddI32Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::AddI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MulI32Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MulI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MinI32Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MinI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MaxI32Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}

void bytecode::MaxI64Op::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto dstReg = getRegisterNumber(getDst());
    const auto lhsReg = getRegisterNumber(getLhs());
    const auto rhsReg = getRegisterNumber(getRhs());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{dstReg, lhsReg, rhsReg});
}
