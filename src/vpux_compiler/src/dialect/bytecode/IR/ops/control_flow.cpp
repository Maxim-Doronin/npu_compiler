//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/bytecode/instructions.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <cstdint>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/control_flow.cpp.inc>

void bytecode::RetOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    writer.appendInstruction(opcode, addrMode, /*operands=*/SmallVector<int16_t>{});
}

void bytecode::AssertOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto opcode = static_cast<uint16_t>(getOpcode());
    const auto addrMode = getAddressingMode();
    const auto conditionReg = getRegisterNumber(getCondition());
    const auto msgSym = getStringIndex(getMsgSym(), getOperation()->getParentOfType<mlir::ModuleOp>());
    writer.appendInstruction(opcode, addrMode, SmallVector<int16_t>{conditionReg, msgSym});
}
