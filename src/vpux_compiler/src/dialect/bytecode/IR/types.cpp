//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/types.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Types.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/bytecode/types.cpp.inc>

void vpux::bytecode::BytecodeDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/bytecode/types.cpp.inc>
            >();
}
