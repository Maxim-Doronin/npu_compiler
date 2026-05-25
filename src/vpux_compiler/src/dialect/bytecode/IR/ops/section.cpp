//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <iterator>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/section.cpp.inc>

void bytecode::FuncSectionOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    getContent().walk([&](bytecode::FuncOp op) {
        op.serialize(writer);
    });
}

size_t bytecode::FuncSectionOp::getBinarySize() {
    size_t size = 0;
    getContent().walk([&](bytecode::FuncOp op) {
        size += op.getBinarySize();
    });
    return size;
}

void bytecode::FuncOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    getBody().walk([&](bytecode::SerializableOpInterface op) {
        op.serialize(writer);
    });
}

size_t bytecode::FuncOp::getBinarySize() {
    size_t size = 0;
    getBody().walk([&](bytecode::SerializableOpInterface op) {
        size += op.getBinarySize();
    });
    return size;
}

mlir::LogicalResult bytecode::FuncOp::verifySymbolUses(mlir::SymbolTableCollection&) {
    auto typeRefName = getFunctionTypeRef();
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
        return emitOpError("expected to be inside a module");
    }

    // Enforce exactly one type section
    auto typeSectionOps = moduleOp.getOps<bytecode::TypeSectionOp>();
    auto numTypeSections = std::distance(typeSectionOps.begin(), typeSectionOps.end());
    if (numTypeSections != 1) {
        return emitOpError("expected exactly one TypeSectionOp in the module, but found ") << numTypeSections;
    }
    auto typeSection = *typeSectionOps.begin();

    // Verify the referenced type exists and is a function type
    for (auto typeOp : typeSection.getContent().getOps<bytecode::TypeOp>()) {
        if (typeOp.getSymName() == typeRefName) {
            if (!mlir::isa<bytecode::FunctionTypeAttr>(typeOp.getValue())) {
                return emitOpError("function type reference '@")
                       << typeRefName << "' resolves to a non-function type in the type section";
            }
            return mlir::success();
        }
    }
    return emitOpError("function type reference '@") << typeRefName << "' could not be resolved in the type section";
}

void bytecode::ConstantSectionOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    getContent().walk([&](bytecode::ConstantOp op) {
        op.serialize(writer);
    });
}

size_t bytecode::ConstantSectionOp::getBinarySize() {
    size_t size = 0;
    getContent().walk([&](bytecode::ConstantOp op) {
        size += op.getBinarySize();
    });
    return size;
}

void bytecode::ConstantOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    auto value = mlir::cast<mlir::DenseElementsAttr>(getValue());
    auto rawdata = value.getRawData();
    writer.appendRawData(reinterpret_cast<const uint8_t*>(rawdata.data()), rawdata.size());
}

size_t bytecode::ConstantOp::getBinarySize() {
    auto value = mlir::cast<mlir::DenseElementsAttr>(getValue());
    return mlir::cast<NDTypeInterface>(value.getType()).getTotalAllocSize().count();
}

mlir::LogicalResult bytecode::ConstantOp::verify() {
    if (!mlir::isa<mlir::DenseElementsAttr>(getValue())) {
        return emitOpError() << "Constant operations only support DenseElementsAttr for values";
    }
    return mlir::success();
}

void bytecode::StringSectionOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    getContent().walk([&](bytecode::StringOp op) {
        op.serialize(writer);
    });
}

size_t bytecode::StringSectionOp::getBinarySize() {
    size_t size = 0;
    getContent().walk([&](bytecode::StringOp op) {
        size += op.getBinarySize();
    });
    return size;
}

void bytecode::StringOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    const auto string = getValue();
    writer.appendRawData(reinterpret_cast<const uint8_t*>(string.data()), string.size());
    writer.appendRawData(reinterpret_cast<const uint8_t*>("\0"), 1);  // Null terminator for string
}

size_t bytecode::StringOp::getBinarySize() {
    return getValue().size() + 1;  // Include null terminator
}

namespace {

uint64_t lookupTypeIndex(mlir::FlatSymbolRefAttr symRef, const llvm::StringMap<uint64_t>& typeIndexMap) {
    auto it = typeIndexMap.find(symRef.getValue());
    VPUX_THROW_WHEN(it == typeIndexMap.end(), "Failed to resolve type symbol reference '{0}'", symRef.getValue());
    return it->second;
}

// Find the parent TypeSectionOp for a TypeOp
bytecode::TypeSectionOp getParentTypeSection(bytecode::TypeOp typeOp) {
    auto parent = typeOp->getParentOp();
    VPUX_THROW_WHEN(parent == nullptr, "TypeOp '{0}' has no parent operation", typeOp.getSymName());
    auto typeSection = mlir::dyn_cast<bytecode::TypeSectionOp>(parent);
    VPUX_THROW_WHEN(typeSection == nullptr, "TypeOp '{0}' parent is not a TypeSectionOp", typeOp.getSymName());
    return typeSection;
}

template <typename T>
void appendValue(vpux::bytecode::BytecodeWriter& writer, T value) {
    writer.appendRawData(reinterpret_cast<const uint8_t*>(&value), sizeof(T));
}

void serializeIntegerType(vpux::bytecode::BytecodeWriter& writer, uint8_t width) {
    appendValue<uint8_t>(writer, 0x01);
    appendValue<uint8_t>(writer, width);
}

void serializeFloatType(vpux::bytecode::BytecodeWriter& writer, uint8_t width, bytecode::FloatFormat format) {
    appendValue<uint8_t>(writer, 0x02);
    appendValue<uint8_t>(writer, width);
    appendValue<uint8_t>(writer, static_cast<uint8_t>(format));
}

constexpr size_t INTEGER_TYPE_BINARY_SIZE = sizeof(uint8_t) * 2;  // id + width
constexpr size_t FLOAT_TYPE_BINARY_SIZE = sizeof(uint8_t) * 3;    // id + width + format

// Serialize a TypeOp using a pre-built type index map (avoids rebuilding per TypeOp)
void serializeTypeOp(bytecode::TypeOp typeOp, vpux::bytecode::BytecodeWriter& writer,
                     const llvm::StringMap<uint64_t>& typeIndexMap) {
    auto value = typeOp.getValue();

    llvm::TypeSwitch<mlir::Attribute>(value)
            .Case<bytecode::IntegerTypeAttr>([&](auto attr) {
                serializeIntegerType(writer, static_cast<uint8_t>(attr.getWidth()));
            })
            .Case<bytecode::FloatTypeAttr>([&](auto attr) {
                serializeFloatType(writer, static_cast<uint8_t>(attr.getWidth()), attr.getFormat());
            })
            .Case<bytecode::OpaqueTypeAttr>([&](auto attr) {
                appendValue<uint8_t>(writer, 0x03);
                appendValue<uint8_t>(writer, static_cast<uint8_t>(attr.getWidth()));
            })
            .Case<bytecode::BufferTypeAttr>([&](auto attr) {
                appendValue<uint8_t>(writer, 0x04);
                appendValue<uint64_t>(writer, lookupTypeIndex(attr.getElementType(), typeIndexMap));
                auto shape = attr.getShape().asArrayRef();
                auto strides = attr.getStrides().asArrayRef();
                appendValue<int64_t>(writer, attr.getRank());
                for (auto dim : shape) {
                    appendValue<int64_t>(writer, dim);
                }
                for (auto stride : strides) {
                    appendValue<int64_t>(writer, stride);
                }
            })
            .Case<bytecode::FunctionTypeAttr>([&](auto attr) {
                appendValue<uint8_t>(writer, 0x05);
                auto args = attr.getArguments();
                auto results = attr.getResults();
                appendValue<uint16_t>(writer, static_cast<uint16_t>(args.size()));
                for (auto argRef : args) {
                    appendValue<uint64_t>(writer,
                                          lookupTypeIndex(mlir::cast<mlir::FlatSymbolRefAttr>(argRef), typeIndexMap));
                }
                appendValue<uint16_t>(writer, static_cast<uint16_t>(results.size()));
                for (auto resRef : results) {
                    appendValue<uint64_t>(writer,
                                          lookupTypeIndex(mlir::cast<mlir::FlatSymbolRefAttr>(resRef), typeIndexMap));
                }
            })
            .Case<mlir::TypeAttr>([&](mlir::TypeAttr typeAttr) {
                auto type = typeAttr.getValue();
                if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
                    serializeIntegerType(writer, static_cast<uint8_t>(intType.getWidth()));
                } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
                    serializeFloatType(writer, static_cast<uint8_t>(floatType.getWidth()),
                                       bytecode::getFloatFormat(floatType));
                } else {
                    VPUX_THROW("Unsupported MLIR type in TypeAttr for serialization in TypeOp '{0}'",
                               typeOp.getSymName());
                }
            })
            .Default([&](auto) {
                VPUX_THROW("Unsupported type attribute for serialization in TypeOp '{0}'", typeOp.getSymName());
            });
}

}  // namespace

void bytecode::TypeSectionOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    auto typeIndexMap = bytecode::buildTypeIndexMap(*this);
    getContent().walk([&](bytecode::TypeOp op) {
        serializeTypeOp(op, writer, typeIndexMap);
    });
}

size_t bytecode::TypeSectionOp::getBinarySize() {
    size_t size = 0;
    getContent().walk([&](bytecode::TypeOp op) {
        size += op.getBinarySize();
    });
    return size;
}

void bytecode::TypeOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    auto typeSection = getParentTypeSection(*this);
    auto typeIndexMap = bytecode::buildTypeIndexMap(typeSection);
    serializeTypeOp(*this, writer, typeIndexMap);
}

size_t bytecode::TypeOp::getBinarySize() {
    auto value = getValue();

    return llvm::TypeSwitch<mlir::Attribute, size_t>(value)
            .Case<bytecode::IntegerTypeAttr>([](auto) -> size_t {
                return INTEGER_TYPE_BINARY_SIZE;
            })
            .Case<bytecode::FloatTypeAttr>([](auto) -> size_t {
                return FLOAT_TYPE_BINARY_SIZE;
            })
            .Case<bytecode::OpaqueTypeAttr>([](auto) -> size_t {
                return sizeof(uint8_t) * 2;  // id + width
            })
            .Case<bytecode::BufferTypeAttr>([](auto attr) -> size_t {
                auto rank = attr.getRank();
                // id + data_type_index + rank + shape[rank] + strides[rank]
                return sizeof(uint8_t) + sizeof(uint64_t) + sizeof(int64_t) +
                       static_cast<size_t>(rank) * sizeof(int64_t) * 2;
            })
            .Case<bytecode::FunctionTypeAttr>([](auto attr) -> size_t {
                auto numArgs = attr.getArguments().size();
                auto numResults = attr.getResults().size();
                // id + num_args + arg_type_indices[numArgs] + num_results + result_type_indices[numResults]
                return sizeof(uint8_t) + sizeof(uint16_t) + numArgs * sizeof(uint64_t) + sizeof(uint16_t) +
                       numResults * sizeof(uint64_t);
            })
            .Case<mlir::TypeAttr>([](mlir::TypeAttr typeAttr) -> size_t {
                auto type = typeAttr.getValue();
                if (mlir::isa<mlir::IntegerType>(type)) {
                    return INTEGER_TYPE_BINARY_SIZE;
                }
                if (mlir::isa<mlir::FloatType>(type)) {
                    return FLOAT_TYPE_BINARY_SIZE;
                }
                VPUX_THROW("Unsupported MLIR type in TypeAttr for size calculation");
            })
            .Default([this](auto) -> size_t {
                VPUX_THROW("Unsupported type attribute for size calculation in TypeOp '{0}'", getSymName());
            });
}

void bytecode::KernelSectionOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    getContent().walk([&](bytecode::KernelOp op) {
        op.serialize(writer);
    });
}

size_t bytecode::KernelSectionOp::getBinarySize() {
    size_t size = 0;
    getContent().walk([&](bytecode::KernelOp op) {
        size += op.getBinarySize();
    });
    return size;
}

void bytecode::KernelOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    auto data = getData();
    writer.appendRawData(reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

size_t bytecode::KernelOp::getBinarySize() {
    return getData().size();
}
