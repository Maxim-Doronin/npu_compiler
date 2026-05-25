//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/bytecode/attributes.cpp.inc>
#include <vpux/compiler/dialect/bytecode/enums.cpp.inc>

//
// Dialect hooks
//

void bytecode::BytecodeDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/bytecode/attributes.cpp.inc>
            >();
}
