//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/StringSet.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <iterator>
#include <memory>
#include <string>

namespace vpux {
#define GEN_PASS_DECL_CONVERTINTERMEDIATEBYTECODEOPS
#define GEN_PASS_DEF_CONVERTINTERMEDIATEBYTECODEOPS
#include "vpux/compiler/dialect/bytecode/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

class SectionOps {
    bytecode::ConstantSectionOp _constantSection;
    bytecode::StringSectionOp _stringSection;
    bytecode::TypeSectionOp _typeSection;

    size_t _nextStringIndex = 0;
    size_t _nextTypeIndex = 0;

    // Deduplication cache: maps a printed type string to the TypeOp symbol name
    llvm::StringMap<std::string> _typeCache;
    // Tracks symbol names already used in the type section to detect collisions
    llvm::StringSet<> _usedSymNames;

    mlir::Location createLoc(mlir::OpBuilder& builder, StringRef sectionName) const {
        return mlir::NameLoc::get(mlir::StringAttr::get(builder.getContext(), sectionName));
    }

    // Create a TypeOp in the type section with the given attribute
    bytecode::TypeOp createTypeOp(const std::string& symName, mlir::Attribute value, mlir::Location loc) {
        auto typeBuilder = mlir::OpBuilder::atBlockEnd(&_typeSection.getContent().getBlocks().front());
        return bytecode::TypeOp::create(typeBuilder, loc, symName, value);
    }

    // Generate a unique key for a type to use for deduplication
    static std::string getTypeKey(mlir::Type type) {
        std::string key;
        llvm::raw_string_ostream os(key);
        type.print(os);
        return key;
    }

public:
    void addConstantSection(mlir::OpBuilder& builder, StringRef name) {
        _constantSection = bytecode::ConstantSectionOp::create(builder, createLoc(builder, name), name);
        _constantSection.getContent().emplaceBlock();
    }
    void addStringSection(mlir::OpBuilder& builder, StringRef name) {
        _stringSection = bytecode::StringSectionOp::create(builder, createLoc(builder, name), name);
        _stringSection.getContent().emplaceBlock();
    }
    void addTypeSection(mlir::OpBuilder& builder, StringRef name) {
        _typeSection = bytecode::TypeSectionOp::create(builder, createLoc(builder, name), name);
        _typeSection.getContent().emplaceBlock();
    }

    bytecode::StringOp addStringToSection(StringRef str, const std::string& prefix, mlir::Location origOpLoc) {
        auto stringBuilder = mlir::OpBuilder::atBlockEnd(&_stringSection.getContent().getBlocks().front());
        auto stringSymName = prefix + "_" + std::to_string(_nextStringIndex++);
        return bytecode::StringOp::create(stringBuilder, origOpLoc, stringSymName, str);
    }

    // Add a type to the type section, with deduplication.
    // Returns the symbol name of the (possibly already existing) TypeOp.
    std::string addTypeToSection(mlir::Type type, mlir::Location loc) {
        auto key = getTypeKey(type);
        auto it = _typeCache.find(key);
        if (it != _typeCache.end()) {
            return it->second;
        }

        std::string symName;
        mlir::Attribute typeAttr;

        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
            auto width = std::to_string(intType.getWidth());
            if (intType.isSigned()) {
                symName = "si" + width;
            } else if (intType.isUnsigned()) {
                symName = "ui" + width;
            } else {
                symName = "i" + width;
            }
            typeAttr = mlir::TypeAttr::get(intType);
        } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
            auto width = static_cast<int64_t>(floatType.getWidth());
            symName = "f" + std::to_string(width);
            auto format = bytecode::getFloatFormat(floatType);
            if (format == bytecode::FloatFormat::BFloat) {
                symName = "bf" + std::to_string(width);
            } else if (format == bytecode::FloatFormat::TFloat) {
                symName = "tf" + std::to_string(width);
            } else if (format == bytecode::FloatFormat::E4M3) {
                symName = "f" + std::to_string(width) + "_e4m3";
            } else if (format == bytecode::FloatFormat::E5M2) {
                symName = "f" + std::to_string(width) + "_e5m2";
            } else if (format == bytecode::FloatFormat::E2M1) {
                symName = "f" + std::to_string(width) + "_e2m1";
            }
            typeAttr = mlir::TypeAttr::get(floatType);
        } else if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            // First, add the element type
            auto elemSymName = addTypeToSection(memrefType.getElementType(), loc);
            auto elemRef = mlir::FlatSymbolRefAttr::get(type.getContext(), elemSymName);

            auto shape = memrefType.getShape();
            int64_t rank = memrefType.getRank();

            // Extract strides and offset using MLIR's built-in utility, which correctly
            // handles both identity layouts and StridedLayoutAttr, including dynamic dimensions
            auto [strides, offset] = memrefType.getStridesAndOffset();
            VPUX_THROW_WHEN(!mlir::ShapedType::isDynamic(offset) && offset != 0,
                            "Bytecode buffer_type cannot encode non-zero offset {0}", offset);

            symName = "buffer_type_" + std::to_string(_nextTypeIndex++);
            auto shapeAttr = mlir::DenseI64ArrayAttr::get(type.getContext(), shape);
            auto stridesAttr = mlir::DenseI64ArrayAttr::get(type.getContext(), strides);
            typeAttr = bytecode::BufferTypeAttr::get(type.getContext(), elemRef, rank, shapeAttr, stridesAttr);
        } else {
            // Fallback: opaque type with 0 width
            symName = "opaque_" + std::to_string(_nextTypeIndex++);
            typeAttr = bytecode::OpaqueTypeAttr::get(type.getContext(), 0);
        }

        // Handle duplicate symbol names (e.g., distinct types that derive the same name)
        std::string baseName = symName;
        while (_usedSymNames.count(symName)) {
            symName = baseName + "_" + std::to_string(_nextTypeIndex++);
        }

        createTypeOp(symName, typeAttr, loc);
        _usedSymNames.insert(symName);
        _typeCache[key] = symName;
        return symName;
    }

    // Decompose a FunctionType into bytecode type attributes and add to type section.
    // Returns the symbol name of the function type entry.
    std::string addFunctionTypeToSection(mlir::FunctionType funcType, mlir::Location loc) {
        auto key = getTypeKey(funcType);
        auto it = _typeCache.find(key);
        if (it != _typeCache.end()) {
            return it->second;
        }

        auto* ctx = funcType.getContext();

        // Add all argument types
        SmallVector<mlir::Attribute> argRefs;
        for (auto argType : funcType.getInputs()) {
            auto symName = addTypeToSection(argType, loc);
            argRefs.push_back(mlir::FlatSymbolRefAttr::get(ctx, symName));
        }

        // Add all result types
        SmallVector<mlir::Attribute> resultRefs;
        for (auto resultType : funcType.getResults()) {
            auto symName = addTypeToSection(resultType, loc);
            resultRefs.push_back(mlir::FlatSymbolRefAttr::get(ctx, symName));
        }

        std::string funcTypeSymName = "function_type_" + std::to_string(_nextTypeIndex++);
        std::string baseName = funcTypeSymName;
        while (_usedSymNames.count(funcTypeSymName)) {
            funcTypeSymName = baseName + "_" + std::to_string(_nextTypeIndex++);
        }

        auto argsAttr = mlir::ArrayAttr::get(ctx, argRefs);
        auto resultsAttr = mlir::ArrayAttr::get(ctx, resultRefs);
        auto funcTypeAttr = bytecode::FunctionTypeAttr::get(ctx, argsAttr, resultsAttr);

        createTypeOp(funcTypeSymName, funcTypeAttr, loc);
        _usedSymNames.insert(funcTypeSymName);
        _typeCache[key] = funcTypeSymName;
        return funcTypeSymName;
    }

    bytecode::ConstantSectionOp getConstantSection() const {
        return _constantSection;
    }
    bytecode::StringSectionOp getStringSection() const {
        return _stringSection;
    }
    bytecode::TypeSectionOp getTypeSection() const {
        return _typeSection;
    }
};

void convertExtAssertOp(bytecode::ExtAssertOp extAssertOp, SectionOps& sections) {
    // Add the assert message to the string section
    auto stringOp = sections.addStringToSection(extAssertOp.getMessage(), "assert_msg", extAssertOp.getLoc());

    // Replace the original ExtAssertOp with AssertOp, which uses the string from the string section
    mlir::OpBuilder builder(extAssertOp);
    builder.setInsertionPoint(extAssertOp);
    auto assertOp =
            builder.create<bytecode::AssertOp>(extAssertOp.getLoc(), extAssertOp.getCondition(), stringOp.getSymName());
    extAssertOp->replaceAllUsesWith(assertOp->getResults());
    extAssertOp.erase();
}

void convertExtFuncOp(bytecode::ExtFuncOp extFuncOp, SectionOps& sections) {
    auto funcType = extFuncOp.getFunctionType();

    // Decompose the function type into bytecode type attributes and add to the type section
    auto funcTypeSymName = sections.addFunctionTypeToSection(funcType, extFuncOp.getLoc());

    // Create the final FuncOp with a symbol reference to the function type in the type section
    mlir::OpBuilder builder(extFuncOp);
    builder.setInsertionPoint(extFuncOp);
    auto funcTypeRef = mlir::FlatSymbolRefAttr::get(extFuncOp.getContext(), funcTypeSymName);
    auto funcOp = bytecode::FuncOp::create(builder, extFuncOp.getLoc(), mlir::TypeRange{}, extFuncOp.getSymNameAttr(),
                                           funcTypeRef);

    // Move the body from the ext.func to the new func
    funcOp.getBody().takeBody(extFuncOp.getBody());
    extFuncOp.erase();
}

}  // namespace

namespace vpux {

class ConvertIntermediateBytecodeOpsPass final :
        public impl::ConvertIntermediateBytecodeOpsBase<ConvertIntermediateBytecodeOpsPass> {
public:
    explicit ConvertIntermediateBytecodeOpsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final {
        auto moduleOp = getOperation();

        const auto funcSection = [&]() -> mlir::FailureOr<bytecode::FuncSectionOp> {
            auto funcSectionOps = moduleOp.getOps<bytecode::FuncSectionOp>();
            const auto numFuncSections = std::distance(funcSectionOps.begin(), funcSectionOps.end());
            if (numFuncSections != 1) {
                _log.error("Expected exactly one FuncSectionOp in the module, but found {0}", numFuncSections);
                return mlir::failure();
            }
            return *funcSectionOps.begin();
        }();
        if (mlir::failed(funcSection)) {
            signalPassFailure();
            return;
        }

        auto sections = prepareSections(*funcSection);

        // Convert ext.func operations first (collect before mutating)
        SmallVector<bytecode::ExtFuncOp> extFuncOps;
        (*funcSection)->walk([&](bytecode::ExtFuncOp extFuncOp) {
            extFuncOps.push_back(extFuncOp);
        });
        for (auto extFuncOp : extFuncOps) {
            convertExtFuncOp(extFuncOp, sections);
        }

        // Convert ext.assert operations inside the (now converted) functions
        SmallVector<bytecode::ExtAssertOp> extAssertOps;
        (*funcSection)->walk([&](bytecode::ExtAssertOp extAssertOp) {
            extAssertOps.push_back(extAssertOp);
        });
        for (auto extAssertOp : extAssertOps) {
            convertExtAssertOp(extAssertOp, sections);
        }
    }

    // Introduce empty sections into the module operation
    SectionOps prepareSections(bytecode::FuncSectionOp funcSection) {
        mlir::OpBuilder builder(funcSection);
        SectionOps sections;
        sections.addConstantSection(builder, bytecode::CONSTANT_SECTION_NAME);
        sections.addStringSection(builder, bytecode::STRING_SECTION_NAME);
        sections.addTypeSection(builder, bytecode::TYPE_SECTION_NAME);
        return sections;
    }
};

}  // namespace vpux

std::unique_ptr<mlir::Pass> vpux::bytecode::createConvertIntermediateBytecodeOpsPass(const Logger& log) {
    return std::make_unique<ConvertIntermediateBytecodeOpsPass>(log);
}
