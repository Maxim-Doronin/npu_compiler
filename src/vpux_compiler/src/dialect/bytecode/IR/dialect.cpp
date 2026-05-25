//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/types.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"

using namespace vpux;

void vpux::bytecode::BytecodeDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/arithmetic.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/control_flow.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/external.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/section.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/bytecode/ops/register.cpp.inc>
            >();
    registerAttributes();
    registerTypes();
}

//
// Generated
//

#include <vpux/compiler/dialect/bytecode/dialect.cpp.inc>
