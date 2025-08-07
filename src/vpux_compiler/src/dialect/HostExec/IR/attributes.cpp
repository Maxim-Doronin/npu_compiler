//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <numeric>

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/HostExec/attributes.cpp.inc>

using namespace vpux;

//
// Dialect hooks
//

void HostExec::HostExecDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/HostExec/attributes.cpp.inc>
            >();
}
