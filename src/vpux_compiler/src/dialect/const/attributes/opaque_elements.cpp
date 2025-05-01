//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/StringExtras.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OperationSupport.h>

using namespace vpux;

//
// OpaqueI64ElementsAttr::print
//

void vpux::Const::OpaqueI64ElementsAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<[";
    const auto values = getValue();
    if (!values.empty()) {
        printer << values[0];
    }
    for (size_t idx = 1; idx < values.size(); idx++) {
        printer << ", ";
        printer << values[idx];
    }
    printer << "]>";
}

//
// OpaqueI64ElementsAttr::parse
//

mlir::Attribute vpux::Const::OpaqueI64ElementsAttr::parse(mlir::AsmParser& parser, mlir::Type type) {
    // This subroutine expects the following format: <[3, 5, 7, 9]>
    // It starts with an angle bracket a.k.a. 'less than' sign.
    if (parser.parseLess()) {
        return nullptr;
    }

    SmallVector<int64_t> data;
    const auto parseI64 = [&]() -> mlir::ParseResult {
        int64_t value = 0;
        if (parser.parseInteger(value)) {
            return mlir::failure();
        }

        data.push_back(value);
        return mlir::success();
    };

    // Parse the [3, 5, 7, 9] part.
    if (parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::Square, parseI64)) {
        return nullptr;
    }

    // Check for an enclosing '>'.
    if (parser.parseGreater()) {
        return nullptr;
    }

    auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type);
    return OpaqueI64ElementsAttr::get(shapedType, ArrayRef(data));
}
