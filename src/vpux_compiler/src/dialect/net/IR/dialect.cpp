//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/dialect.hpp>
#include <vpux/compiler/dialect/net/IR/dialect.hpp>
#include <vpux/compiler/dialect/net/IR/ops.hpp>

using namespace vpux;

void vpux::net::NetDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/net/ops.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/net/dialect.cpp.inc>
