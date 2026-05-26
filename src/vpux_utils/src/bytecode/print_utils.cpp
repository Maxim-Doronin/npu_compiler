//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/print_utils.hpp"
#include "vpux/utils/bytecode/instructions.hpp"
#include "vpux/utils/bytecode/section_header_table.hpp"
#include "vpux/utils/bytecode/span.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace vpux;

namespace {

bool checkInstructionInBounds(bytecode::Span<uint8_t> functionBody, bytecode::OpCode opcode) {
    const auto instructionSizeIt = bytecode::INSTRUCTION_SIZES.find(opcode);
    if (instructionSizeIt == bytecode::INSTRUCTION_SIZES.end()) {
        std::cerr << "Error: Unknown opcode " << static_cast<uint16_t>(opcode) << std::endl;
        return false;
    }
    const auto instructionSize = instructionSizeIt->second;
    if (functionBody.size() < instructionSize) {
        std::cerr << "Error: Reached end of function body while decoding an instruction. "
                  << "This likely means the function body is malformed." << std::endl;
        return false;
    }
    return true;
}

void printFunctionBody(bytecode::Span<uint8_t> body, size_t indentLevel = 0) {
    const auto printInstruction = [&](std::string_view opcode, const std::vector<int64_t>& operands = {}) {
        bytecode::printIndent(indentLevel);
        std::cout << opcode;
        for (size_t i = 0; i < operands.size(); ++i) {
            std::cout << " " << operands[i];
            if (i < operands.size() - 1) {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
    };

    while (body.begin() != nullptr && body.size() > 0) {
        const auto opcode = static_cast<bytecode::OpCode>(bytecode::getOpcode(body.begin()));
        if (!checkInstructionInBounds(body, opcode)) {
            break;
        }
        switch (opcode) {
        case bytecode::OpCode::ADD_I32: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcReg1Num = bytecode::getOperand(body.begin(), 1);
            const auto srcReg2Num = bytecode::getOperand(body.begin(), 2);
            printInstruction("add.i32", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case bytecode::OpCode::ADD_I64: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcReg1Num = bytecode::getOperand(body.begin(), 1);
            const auto srcReg2Num = bytecode::getOperand(body.begin(), 2);
            printInstruction("add.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case bytecode::OpCode::MUL_I32: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcReg1Num = bytecode::getOperand(body.begin(), 1);
            const auto srcReg2Num = bytecode::getOperand(body.begin(), 2);
            printInstruction("mul.i32", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case bytecode::OpCode::MUL_I64: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcReg1Num = bytecode::getOperand(body.begin(), 1);
            const auto srcReg2Num = bytecode::getOperand(body.begin(), 2);
            printInstruction("mul.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case bytecode::OpCode::MIN_I32: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcReg1Num = bytecode::getOperand(body.begin(), 1);
            const auto srcReg2Num = bytecode::getOperand(body.begin(), 2);
            printInstruction("min.i32", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case bytecode::OpCode::MIN_I64: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcReg1Num = bytecode::getOperand(body.begin(), 1);
            const auto srcReg2Num = bytecode::getOperand(body.begin(), 2);
            printInstruction("min.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case bytecode::OpCode::MAX_I32: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcReg1Num = bytecode::getOperand(body.begin(), 1);
            const auto srcReg2Num = bytecode::getOperand(body.begin(), 2);
            printInstruction("max.i32", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case bytecode::OpCode::MAX_I64: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcReg1Num = bytecode::getOperand(body.begin(), 1);
            const auto srcReg2Num = bytecode::getOperand(body.begin(), 2);
            printInstruction("max.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case bytecode::OpCode::RET: {
            printInstruction("ret");
            break;
        }
        case bytecode::OpCode::SET: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto srcRegNum = bytecode::getOperand(body.begin(), 1);
            printInstruction("set", {dstRegNum, srcRegNum});
            break;
        }
        case bytecode::OpCode::SET_IMM: {
            const auto dstRegNum = bytecode::getOperand(body.begin(), 0);
            const auto immValue = bytecode::get64BitImm(body.begin(), 1);
            printInstruction("set.imm", {dstRegNum, immValue});
            break;
        }
        case bytecode::OpCode::ASSERT: {
            const auto conditionRegNum = bytecode::getOperand(body.begin(), 0);
            const auto msgSymIndex = bytecode::getOperand(body.begin(), 1);
            printInstruction("assert", {conditionRegNum, msgSymIndex});
            break;
        }
        default:
            std::cerr << "Unknown opcode: " << static_cast<uint16_t>(opcode) << std::endl;
            return;
        }
        const auto instructionSize = bytecode::INSTRUCTION_SIZES.at(opcode);
        body = bytecode::Span<uint8_t>(body.begin() + instructionSize, body.size() - instructionSize);
    }
}

void printConstant(bytecode::Span<uint8_t> constant, bool printFull) {
    const auto printEntireConstant = [&]() {
        for (const auto& byte : constant) {
            std::cout << bytecode::toHex(byte);
        }
    };

    const auto printPartialConstant = [&]() {
        constexpr size_t numBytesToPrintBeforeAndAfter = 4;
        if (constant.size() <= numBytesToPrintBeforeAndAfter * 2) {
            printEntireConstant();
            return;
        }
        for (size_t i = 0; i < numBytesToPrintBeforeAndAfter; ++i) {
            std::cout << bytecode::toHex(constant[i]);
        }
        std::cout << "...";
        for (size_t i = constant.size() - numBytesToPrintBeforeAndAfter; i < constant.size(); ++i) {
            std::cout << bytecode::toHex(constant[i]);
        }
    };

    if (constant.size() == 0) {
        std::cout << "(empty)";
        return;
    }
    std::cout << "0x";
    if (printFull) {
        printEntireConstant();
    } else {
        // Only print the first and last few elements when printFull is disabled
        printPartialConstant();
    }
}

void printString(bytecode::Span<uint8_t> string) {
    if (string.size() == 0) {
        std::cout << "\"\"";
        return;
    }
    for (const auto& byte : string) {
        if (byte == '\0') {
            // Ensure the null terminator character is printed as well
            std::cout << "\\0";
            continue;
        }
        std::cout << byte;
    }
}

void printType(bytecode::Span<uint8_t> type) {
    // TODO: Implement type printing once they are serialized
    (void)type;
}

}  // namespace

std::string bytecode::toHex(uint8_t byte) {
    std::stringstream ss;
    ss << std::hex << std::uppercase << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
    return ss.str();
};

void bytecode::printFunctionSection(const bytecode::SectionHeader& sectionHeader,
                                    bytecode::Span<uint8_t> sectionContent, size_t sectionIdx, size_t indentLevel) {
    bytecode::printIndent(indentLevel);
    std::cout << "Function section " << sectionIdx << std::endl;

    const auto funcSectionInfo = dynamic_cast<vpux::bytecode::details::FunctionSectionInfo*>(sectionHeader.info.get());
    if (funcSectionInfo == nullptr) {
        std::cerr << "Error: Function section header has invalid info type" << std::endl;
        return;
    }
    for (const auto& funcInfo : funcSectionInfo->functionInfos) {
        bytecode::printIndent(indentLevel + 1);
        // TODO: Get name from string section using nameIndex
        std::cout << "Function name: " << funcInfo.nameIndex << std::endl;
        if (funcInfo.bodyOffset + funcInfo.bodySize > sectionContent.size()) {
            std::cerr << "Error: Section of type " << bytecode::getSectionTypeString(sectionHeader.type)
                      << " has invalid offset and size that exceed section buffer" << std::endl;
            continue;
        }
        const auto body = bytecode::Span<uint8_t>(sectionContent.begin() + funcInfo.bodyOffset, funcInfo.bodySize);
        printFunctionBody(body, indentLevel + 2);
    }
}

void bytecode::printDataSection(const bytecode::SectionHeader& sectionHeader, bytecode::Span<uint8_t> sectionContent,
                                size_t sectionIdx, bytecode::SectionType sectionType, bool printFull,
                                size_t indentLevel) {
    bytecode::printIndent(indentLevel);
    std::cout << bytecode::getSectionTypeString(sectionType) << " section " << sectionIdx << std::endl;

    const auto dataSectionInfo = dynamic_cast<vpux::bytecode::details::DataSectionInfo*>(sectionHeader.info.get());
    if (dataSectionInfo == nullptr) {
        std::cerr << "Error: Section header has invalid info type" << std::endl;
        return;
    }
    for (size_t cstIdx = 0; cstIdx < dataSectionInfo->dataInfos.size(); ++cstIdx) {
        const auto& dataInfo = dataSectionInfo->dataInfos[cstIdx];
        bytecode::printIndent(indentLevel + 1);
        std::cout << bytecode::getSectionTypeString(sectionType) << " " << cstIdx << ": ";
        if (dataInfo.offset + dataInfo.size > sectionContent.size()) {
            std::cerr << "Error: Section of type " << bytecode::getSectionTypeString(sectionType)
                      << " has invalid offset and size that exceed section buffer" << std::endl;
            continue;
        }
        const auto content = bytecode::Span<uint8_t>(sectionContent.begin() + dataInfo.offset, dataInfo.size);
        if (sectionType == bytecode::SectionType::ConstantSection) {
            printConstant(content, printFull);
        } else if (sectionType == bytecode::SectionType::KernelSection) {
            printConstant(content, printFull);
        } else if (sectionType == bytecode::SectionType::StringSection) {
            printString(content);
        } else if (sectionType == bytecode::SectionType::TypeSection) {
            printType(content);
        } else {
            std::cerr << "Error: Unsupported section type for body section printing: "
                      << bytecode::getSectionTypeString(sectionType) << std::endl;
        }
        std::cout << std::endl;
    }
}
