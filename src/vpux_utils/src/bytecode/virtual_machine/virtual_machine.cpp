//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/virtual_machine/virtual_machine.hpp"
#include "vpux/utils/bytecode/bytecode_reader.hpp"
#include "vpux/utils/bytecode/instructions.hpp"
#include "vpux/utils/bytecode/section_header_table.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define DEBUG_MODE 1

#ifdef DEBUG_MODE
#define DEBUG_LOG(x) std::cout << x;
#else
#define DEBUG_LOG(x)
#endif

using namespace vpux;

namespace {

inline bool checkPCInBounds(const uint8_t* pc, bytecode::OpCode opcode, const uint8_t* functionBodyEnd) {
    const auto instructionSizeIt = bytecode::INSTRUCTION_SIZES.find(opcode);
    if (instructionSizeIt == bytecode::INSTRUCTION_SIZES.end()) {
        std::cerr << "Error: Unknown opcode " << static_cast<uint16_t>(opcode) << std::endl;
        return false;
    }
    const auto instructionSize = instructionSizeIt->second;
    if (pc + instructionSize > functionBodyEnd) {
        std::cerr << "Error: Reached end of function body while decoding an instruction. "
                  << "This likely means the function body is malformed." << std::endl;
        return false;
    }
    return true;
}

}  // namespace

bytecode::Function::Function(std::string name, uint64_t numGeneralRegisters, bool isEntrypoint,
                             std::vector<uint8_t> body)
        : _name(std::move(name)),
          _numGeneralRegisters(numGeneralRegisters),
          _isEntrypoint(isEntrypoint),
          _body(std::move(body)) {
}

std::string bytecode::Function::getName() const {
    return _name;
}

uint64_t bytecode::Function::getNumGeneralRegisters() const {
    return _numGeneralRegisters;
}

bool bytecode::Function::isEntrypoint() const {
    return _isEntrypoint;
}

const std::vector<uint8_t>& bytecode::Function::getBody() const {
    return _body;
}

bytecode::CallFrame::CallFrame(uint64_t numRegisters, const uint8_t* returnAddress)
        : _registers(numRegisters, 0), _returnAddress(returnAddress) {
}

int64_t& bytecode::CallFrame::getReg(int16_t index) {
    if (index >= static_cast<int16_t>(_registers.size())) {
        throw std::out_of_range("Register index out of range");
    }
    return _registers[index];
}

void bytecode::CallFrame::setReg(int16_t index, int64_t value) {
    if (index >= static_cast<int16_t>(_registers.size())) {
        throw std::out_of_range("Register index out of range");
    }
    _registers[index] = value;
}

const uint8_t* bytecode::CallFrame::getReturnAddress() const {
    return _returnAddress;
}

bool bytecode::VirtualMachine::parse(const std::vector<uint8_t>& bytecode) {
    vpux::bytecode::BytecodeReader reader(bytecode);
    if (!reader.parseFile()) {
        std::cerr << "Error: Failed to parse bytecode file." << std::endl;
        return false;
    }

    auto& sectionHeaderTable = reader.getSectionHeaderTable();
    auto& sectionHeaders = sectionHeaderTable.getSectionHeaders();
    auto& sections = reader.getSections();

    // Iterate over section headers to find function sections and extract individual function bodies
    for (size_t headerIdx = 0; headerIdx < sectionHeaders.size(); ++headerIdx) {
        const auto& header = sectionHeaders[headerIdx];
        if (header.type != vpux::bytecode::SectionType::FuncSection) {
            continue;
        }
        auto funcSectionInfo = dynamic_cast<vpux::bytecode::details::FunctionSectionInfo*>(header.info.get());
        if (funcSectionInfo == nullptr) {
            std::cerr << "Error: Function section header does not contain function section info" << std::endl;
            return false;
        }
        for (size_t i = 0; i < funcSectionInfo->functionInfos.size(); ++i) {
            if (headerIdx >= sections.size()) {
                std::cerr << "Error: Could not find associated section with section header " << headerIdx << std::endl;
                return false;
            }
            const auto& funcInfo = funcSectionInfo->functionInfos[i];
            const auto functionName = std::to_string(funcInfo.nameIndex);  // TODO: Get name from string section
            const auto numGeneralRegisters = funcInfo.numGeneralRegisters;
            const auto isEntrypoint = funcSectionInfo->entrypointFunctionIndex == i;
            const auto& section = sections[headerIdx];
            const auto bodyStart = section.begin() + static_cast<int64_t>(funcInfo.bodyOffset);
            const auto bodyEnd = bodyStart + static_cast<int64_t>(funcInfo.bodySize);
            if (bodyEnd > section.end()) {
                std::cerr << "Error: Function body exceeds section bounds for function " << functionName << std::endl;
                return false;
            }
            std::vector<uint8_t> functionBody(bodyStart, bodyEnd);
            _functions.emplace_back(functionName, numGeneralRegisters, isEntrypoint, std::move(functionBody));
        }
    }

    return true;
}

bool bytecode::VirtualMachine::print(const std::vector<uint8_t>& bytecode, bool printFull, size_t indentLevel) const {
    vpux::bytecode::BytecodeReader reader(bytecode);
    if (!reader.parseFile()) {
        std::cerr << "Error: Failed to parse bytecode file." << std::endl;
        return false;
    }
    reader.printFile(printFull, indentLevel);
    return true;
}

void bytecode::VirtualMachine::run() {
    auto entrypointFunctionIt = std::find_if(_functions.begin(), _functions.end(), [](const Function& func) {
        return func.isEntrypoint();
    });
    if (entrypointFunctionIt == _functions.end()) {
        std::cout << "Error: No entrypoint function found. Nothing to execute" << std::endl;
        return;
    }

    const auto& entrypointFunction = *entrypointFunctionIt;
    std::cout << "Executing entrypoint function: " << entrypointFunction.getName() << std::endl;

    _state = State::Running;
    execute(entrypointFunction);
}

void bytecode::VirtualMachine::incrementPC(bytecode::OpCode opcode) {
    if (_state != State::Running) {
        return;
    }
    const auto instructionSize = bytecode::INSTRUCTION_SIZES.at(opcode);
    _pc += instructionSize;
    DEBUG_LOG("  New PC: " << static_cast<const void*>(_pc) << std::endl);
}

void bytecode::VirtualMachine::execute(const Function& function) {
    _pc = function.getBody().data();

    DEBUG_LOG("Executing function " << function.getName() << " with body size: " << function.getBody().size()
                                    << " bytes" << std::endl);
    DEBUG_LOG("  Initial PC: " << static_cast<const void*>(_pc) << std::endl);

    // The entrypoint's return address is nullptr, as there is no function to return to
    constexpr uint8_t* exitReturnAddr = nullptr;
    CallFrame frame(function.getNumGeneralRegisters(), exitReturnAddr);

    // Main dispatch loop: decode opcode at the current program counter, execute the corresponding operation, then
    // advance the PC by the instruction size. Loop exits when state transitions away from Running.
    const auto endOfFunction = function.getBody().data() + function.getBody().size();
    while (_state == State::Running && _pc < endOfFunction) {
        const auto opcode = static_cast<OpCode>(getOpcode(_pc));
        if (!checkPCInBounds(_pc, opcode, endOfFunction)) {
            _state = State::Halted;
            break;
        }

        switch (opcode) {
        case OpCode::ADD_I32: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcReg1Num = getOperand(_pc, 1);
            const auto srcReg2Num = getOperand(_pc, 2);
            frame.setReg(dstRegNum, static_cast<int32_t>(frame.getReg(srcReg1Num)) +
                                            static_cast<int32_t>(frame.getReg(srcReg2Num)));
            DEBUG_LOG("  add.i32 " << dstRegNum << ", " << srcReg1Num << ", " << srcReg2Num << " ("
                                   << static_cast<int32_t>(frame.getReg(srcReg1Num)) << " + "
                                   << static_cast<int32_t>(frame.getReg(srcReg2Num)) << " = "
                                   << static_cast<int32_t>(frame.getReg(dstRegNum)) << ")" << std::endl);
            break;
        }
        case OpCode::ADD_I64: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcReg1Num = getOperand(_pc, 1);
            const auto srcReg2Num = getOperand(_pc, 2);
            frame.setReg(dstRegNum, frame.getReg(srcReg1Num) + frame.getReg(srcReg2Num));
            DEBUG_LOG("  add.i64 " << dstRegNum << ", " << srcReg1Num << ", " << srcReg2Num << " ("
                                   << frame.getReg(srcReg1Num) << " + " << frame.getReg(srcReg2Num) << " = "
                                   << frame.getReg(dstRegNum) << ")" << std::endl);
            break;
        }
        case OpCode::SET: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcRegNum = getOperand(_pc, 1);
            frame.setReg(dstRegNum, frame.getReg(srcRegNum));
            DEBUG_LOG("  set " << dstRegNum << ", " << srcRegNum << " (reg[" << dstRegNum
                               << "] = " << frame.getReg(srcRegNum) << ")" << std::endl);
            break;
        }
        case OpCode::SET_IMM: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto immValue = get64BitImm(_pc, 1);
            frame.setReg(dstRegNum, immValue);
            DEBUG_LOG("  set.imm " << dstRegNum << ", " << immValue << " (reg[" << dstRegNum << "] = " << immValue
                                   << ")" << std::endl);
            break;
        }
        case OpCode::ASSERT: {
            const auto conditionRegNum = getOperand(_pc, 0);
            const auto msgSymIndex = getOperand(_pc, 1);
            // Note: The message symbol index will be replaced with the actual message once the string section is used
            // by the VM
            if (frame.getReg(conditionRegNum) == 0) {
                std::cerr << "Assertion failed: " << " (msgSymIndex: " << msgSymIndex << ")" << std::endl;
                _state = State::Halted;
            } else {
                DEBUG_LOG("  assert " << conditionRegNum << ", " << msgSymIndex << std::endl);
            }
            break;
        }
        case OpCode::MUL_I32: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcReg1Num = getOperand(_pc, 1);
            const auto srcReg2Num = getOperand(_pc, 2);
            frame.setReg(dstRegNum, static_cast<int32_t>(frame.getReg(srcReg1Num)) *
                                            static_cast<int32_t>(frame.getReg(srcReg2Num)));
            DEBUG_LOG("  mul.i32 " << dstRegNum << ", " << srcReg1Num << ", " << srcReg2Num << " ("
                                   << static_cast<int32_t>(frame.getReg(srcReg1Num)) << " * "
                                   << static_cast<int32_t>(frame.getReg(srcReg2Num)) << " = "
                                   << static_cast<int32_t>(frame.getReg(dstRegNum)) << ")" << std::endl);
            break;
        }
        case OpCode::MUL_I64: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcReg1Num = getOperand(_pc, 1);
            const auto srcReg2Num = getOperand(_pc, 2);
            frame.setReg(dstRegNum, frame.getReg(srcReg1Num) * frame.getReg(srcReg2Num));
            DEBUG_LOG("  mul.i64 " << dstRegNum << ", " << srcReg1Num << ", " << srcReg2Num << " ("
                                   << frame.getReg(srcReg1Num) << " * " << frame.getReg(srcReg2Num) << " = "
                                   << frame.getReg(dstRegNum) << ")" << std::endl);
            break;
        }
        case OpCode::MIN_I32: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcReg1Num = getOperand(_pc, 1);
            const auto srcReg2Num = getOperand(_pc, 2);
            frame.setReg(dstRegNum, std::min(static_cast<int32_t>(frame.getReg(srcReg1Num)),
                                             static_cast<int32_t>(frame.getReg(srcReg2Num))));
            DEBUG_LOG("  min.i32 " << dstRegNum << ", " << srcReg1Num << ", " << srcReg2Num << " ("
                                   << static_cast<int32_t>(frame.getReg(srcReg1Num)) << " min "
                                   << static_cast<int32_t>(frame.getReg(srcReg2Num)) << " = "
                                   << static_cast<int32_t>(frame.getReg(dstRegNum)) << ")" << std::endl);
            break;
        }
        case OpCode::MIN_I64: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcReg1Num = getOperand(_pc, 1);
            const auto srcReg2Num = getOperand(_pc, 2);
            frame.setReg(dstRegNum, std::min(frame.getReg(srcReg1Num), frame.getReg(srcReg2Num)));
            DEBUG_LOG("  min.i64 " << dstRegNum << ", " << srcReg1Num << ", " << srcReg2Num << " ("
                                   << frame.getReg(srcReg1Num) << " min " << frame.getReg(srcReg2Num) << " = "
                                   << frame.getReg(dstRegNum) << ")" << std::endl);
            break;
        }
        case OpCode::MAX_I32: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcReg1Num = getOperand(_pc, 1);
            const auto srcReg2Num = getOperand(_pc, 2);
            frame.setReg(dstRegNum, std::max(static_cast<int32_t>(frame.getReg(srcReg1Num)),
                                             static_cast<int32_t>(frame.getReg(srcReg2Num))));
            DEBUG_LOG("  max.i32 " << dstRegNum << ", " << srcReg1Num << ", " << srcReg2Num << " ("
                                   << static_cast<int32_t>(frame.getReg(srcReg1Num)) << " max "
                                   << static_cast<int32_t>(frame.getReg(srcReg2Num)) << " = "
                                   << static_cast<int32_t>(frame.getReg(dstRegNum)) << ")" << std::endl);
            break;
        }
        case OpCode::MAX_I64: {
            const auto dstRegNum = getOperand(_pc, 0);
            const auto srcReg1Num = getOperand(_pc, 1);
            const auto srcReg2Num = getOperand(_pc, 2);
            frame.setReg(dstRegNum, std::max(frame.getReg(srcReg1Num), frame.getReg(srcReg2Num)));
            DEBUG_LOG("  max.i64 " << dstRegNum << ", " << srcReg1Num << ", " << srcReg2Num << " ("
                                   << frame.getReg(srcReg1Num) << " max " << frame.getReg(srcReg2Num) << " = "
                                   << frame.getReg(dstRegNum) << ")" << std::endl);
            break;
        }
        case OpCode::RET: {
            if (frame.getReturnAddress() == nullptr) {
                DEBUG_LOG("  ret (returning from entrypoint, finishing execution)" << std::endl);
                _state = State::Finalized;
            } else {
                // In a more complete implementation, we would set the PC to the return address and pop the call frame
                std::cerr << "Error: RET instruction encountered in non-entrypoint function. "
                          << "Nested function calls are not supported in this initial implementation." << std::endl;
                _state = State::Halted;
            }
            break;
        }
        default:
            std::cerr << "Error: Unknown opcode " << static_cast<uint16_t>(opcode) << std::endl;
            _state = State::Halted;
            break;
        }
        incrementPC(opcode);
    }
}
