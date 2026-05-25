//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/dialect.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"

namespace vpux {
namespace type {

QuantileDialect::~QuantileDialect() = default;

void QuantileDialect::initialize() {
    registerTypes();
}

void QuantileDialect::registerTypes() {
    addTypes<vpux::type::NF4Type, vpux::type::QuantileType>();
}

mlir::OptionalParseResult QuantileDialect::generatedTypeParser(mlir::AsmParser& parser, llvm::StringRef* mnemonic,
                                                               mlir::Type& value) {
    return mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
            .Case(vpux::type::QuantileType::getMnemonic(),
                  [&](llvm::StringRef, llvm::SMLoc) {
                      value = vpux::type::QuantileType::parse(parser);
                      return mlir::success(!!value);
                  })
            .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
                *mnemonic = keyword;
                return std::nullopt;
            });
}

mlir::LogicalResult QuantileDialect::generatedTypePrinter(mlir::Type def, mlir::AsmPrinter& printer) {
    return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(def)
            .Case<vpux::type::QuantileType>([&](auto t) {
                printer << vpux::type::QuantileType::getMnemonic();
                t.print(printer);
                return ::mlir::success();
            })
            .Default([](auto) {
                return ::mlir::failure();
            });
}

mlir::Type QuantileDialect::parseType(mlir::DialectAsmParser& parser) const {
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    llvm::StringRef mnemonic;
    mlir::Type genType;
    auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
    if (parseResult.has_value()) {
        return genType;
    }

    parser.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `" << getNamespace() << "`";
    return {};
}

/// Print a type registered to this dialect.
void QuantileDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
    if (mlir::succeeded(generatedTypePrinter(type, printer))) {
        return;
    }
};

}  // namespace type
}  // namespace vpux
