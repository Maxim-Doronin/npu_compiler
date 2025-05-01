//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"

//
// initialize
//

void vpux::NPUReg40XX::NPUReg40XXDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.cpp.inc>
            >();
    registerTypes();
    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.cpp.inc>
