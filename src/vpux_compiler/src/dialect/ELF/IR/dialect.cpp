//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"

//
// initialize
//

void vpux::ELF::ELFDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/ELF/ops.cpp.inc>
            >();

    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/ELF/dialect.cpp.inc>
