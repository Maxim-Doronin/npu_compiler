//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/bytecode/magic_number.hpp"
#include "vpux/utils/bytecode/section_header_table.hpp"
#include "vpux/utils/bytecode/serialization_utils.hpp"
#include "vpux/utils/bytecode/version.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

using namespace vpux;

namespace {

uint64_t countGeneralRegisters(bytecode::FuncOp funcOp) {
    auto registerOps = funcOp.getOps<bytecode::GeneralRegisterOp>();
    std::optional<int64_t> maxRegNum;
    for (auto regOp : registerOps) {
        if (!maxRegNum.has_value() || regOp.getRegNum() > maxRegNum.value()) {
            maxRegNum = regOp.getRegNum();
        }
    }
    const auto numGeneralRegisters = maxRegNum.has_value() ? maxRegNum.value() + 1 : 0;
    return numGeneralRegisters;
}

// Parse a function section and create a corresponding section header
bytecode::SectionHeader parseFunctionSectionHeader(bytecode::FuncSectionOp funcSection,
                                                   bytecode::TypeSectionOp typeSection) {
    bytecode::SectionHeader header{};
    header.type = bytecode::SectionType::FuncSection;
    header.nameIndex = 0;  // Placeholder for actual name index calculation
    header.offset = 0;     // Placeholder for actual offset calculation
    header.size = funcSection.getBinarySize();

    auto typeIndexMap = bytecode::buildTypeIndexMap(typeSection);

    bytecode::details::FunctionSectionInfo funcInfo;
    funcInfo.entrypointFunctionIndex = 0;  // Placeholder for actual entry point index calculation
    auto functionOps = funcSection.getContent().getOps<bytecode::FuncOp>();
    funcInfo.numFunctions = std::distance(functionOps.begin(), functionOps.end());
    size_t bodyOffset = 0;
    for (auto funcOp : functionOps) {
        auto typeRefName = funcOp.getFunctionTypeRef();
        auto it = typeIndexMap.find(typeRefName);
        VPUX_THROW_WHEN(it == typeIndexMap.end(),
                        "Failed to resolve function type reference '@{0}' in the type section", typeRefName);

        bytecode::details::FunctionSectionInfo::FunctionInfo info{};
        info.nameIndex = 0;  // Placeholder for actual function name index calculation
        info.functionTypeIndex = it->second;
        info.numGeneralRegisters = countGeneralRegisters(funcOp);
        info.bodyOffset = bodyOffset;
        info.bodySize = funcOp.getBinarySize();
        funcInfo.functionInfos.push_back(info);
        bodyOffset += info.bodySize;
    }
    header.info = std::make_unique<bytecode::details::FunctionSectionInfo>(funcInfo);
    return header;
}

// Parse a section that contains only data (i.e. offsets and sizes) and create a corresponding section header
// This is used for ConstantSection, StringSection and TypeSection
template <typename DataSectionOp, typename DataOp>
bytecode::SectionHeader parseDataSectionHeader(DataSectionOp sectionOp, bytecode::SectionType sectionType) {
    bytecode::SectionHeader header{};
    header.type = sectionType;
    header.nameIndex = 0;  // Placeholder for actual name index calculation
    header.offset = 0;     // Placeholder for actual offset calculation
    header.size = sectionOp.getBinarySize();

    bytecode::details::DataSectionInfo dataInfo;
    auto dataOps = sectionOp.getContent().template getOps<DataOp>();
    dataInfo.numData = std::distance(dataOps.begin(), dataOps.end());
    size_t dataOffset = 0;
    for (auto dataOp : dataOps) {
        bytecode::details::DataSectionInfo::DataInfo info{};
        info.offset = dataOffset;
        info.size = dataOp.getBinarySize();
        dataInfo.dataInfos.push_back(info);
        dataOffset += info.size;
    }
    header.info = std::make_unique<bytecode::details::DataSectionInfo>(dataInfo);
    return header;
}

}  // namespace

bytecode::BytecodeWriter::BytecodeWriter(mlir::ModuleOp moduleOp)
        : _moduleOp(moduleOp), _bytecodeBuffer(), _sectionHeaderTable() {
    prepareSectionHeaderTable();
}

void bytecode::BytecodeWriter::prepareSectionHeaderTable() {
    // Enforce exactly one type section when function sections are present
    auto typeSectionOps = _moduleOp.getOps<bytecode::TypeSectionOp>();
    auto numTypeSections = std::distance(typeSectionOps.begin(), typeSectionOps.end());
    VPUX_THROW_UNLESS(numTypeSections <= 1, "Expected at most one TypeSectionOp in the module, but found {0}",
                      numTypeSections);

    _moduleOp.walk([&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*>(op)
                .Case<bytecode::FuncSectionOp>([&](bytecode::FuncSectionOp funcSection) {
                    VPUX_THROW_WHEN(numTypeSections == 0,
                                    "FuncSectionOp requires a TypeSectionOp for function type resolution");
                    _sectionHeaderTable.addSectionHeader(
                            parseFunctionSectionHeader(funcSection, *typeSectionOps.begin()));
                })
                .Case<bytecode::ConstantSectionOp>([&](bytecode::ConstantSectionOp constantSection) {
                    _sectionHeaderTable.addSectionHeader(parseDataSectionHeader<ConstantSectionOp, ConstantOp>(
                            constantSection, SectionType::ConstantSection));
                })
                .Case<bytecode::KernelSectionOp>([&](bytecode::KernelSectionOp kernelSection) {
                    _sectionHeaderTable.addSectionHeader(parseDataSectionHeader<KernelSectionOp, KernelOp>(
                            kernelSection, SectionType::KernelSection));
                })
                .Case<bytecode::StringSectionOp>([&](bytecode::StringSectionOp stringSection) {
                    _sectionHeaderTable.addSectionHeader(parseDataSectionHeader<StringSectionOp, StringOp>(
                            stringSection, SectionType::StringSection));
                })
                .Case<bytecode::TypeSectionOp>([&](bytecode::TypeSectionOp typeSection) {
                    _sectionHeaderTable.addSectionHeader(
                            parseDataSectionHeader<TypeSectionOp, TypeOp>(typeSection, SectionType::TypeSection));
                })
                .Default([](mlir::Operation*) {});
    });

    _sectionHeaderTable.computeOffsets();
}

void bytecode::BytecodeWriter::appendFileHeader() {
    MagicNumber magicNumber(MAGIC_NUMBER);
    magicNumber.appendTo(_bytecodeBuffer);

    bytecode::Version version(1, 0, 0);  // Placeholder for actual target version
    version.appendTo(_bytecodeBuffer);

    _sectionHeaderTable.appendTo(_bytecodeBuffer);
}

void bytecode::BytecodeWriter::appendSections() {
    _moduleOp.walk([&](bytecode::FuncSectionOp funcSection) {
        funcSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::ConstantSectionOp constantSection) {
        constantSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::KernelSectionOp kernelSection) {
        kernelSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::StringSectionOp stringSection) {
        stringSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::TypeSectionOp typeSection) {
        typeSection.serialize(*this);
    });
}

void bytecode::BytecodeWriter::appendInstruction(uint16_t opcode, uint16_t addressingMode, ArrayRef<int16_t> operands) {
    opcode |= addressingMode;  // Embed the addressing mode into the opcode
    appendValueTo(_bytecodeBuffer, opcode);

    const auto operandsData = reinterpret_cast<const uint8_t*>(operands.data());
    _bytecodeBuffer.insert(_bytecodeBuffer.end(), operandsData, operandsData + operands.size() * sizeof(int16_t));
}

void bytecode::BytecodeWriter::appendInstruction(uint16_t opcode, uint16_t addressingMode,
                                                 ArrayRef<uint8_t> binaryOperands) {
    opcode |= addressingMode;  // Embed the addressing mode into the opcode
    appendValueTo(_bytecodeBuffer, opcode);

    _bytecodeBuffer.insert(_bytecodeBuffer.end(), binaryOperands.begin(), binaryOperands.end());
}

void bytecode::BytecodeWriter::appendRawData(const uint8_t* data, size_t size) {
    _bytecodeBuffer.insert(_bytecodeBuffer.end(), data, data + size);
}

void bytecode::BytecodeWriter::writeTo(llvm::raw_ostream& os) {
    os.write(reinterpret_cast<const char*>(_bytecodeBuffer.data()), _bytecodeBuffer.size());
    os.flush();
}
