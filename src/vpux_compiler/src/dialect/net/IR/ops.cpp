//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/net/ops.cpp.inc>

namespace vpux::net {
void DataInfoOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::StringRef name,
                       mlir::Type userType) {
    build(builder, state, name, userType, /*originalShape=*/nullptr, /*friendlyName=*/nullptr, /*inputName=*/nullptr,
          /*tensorNames=*/nullptr, /*sectionsCount=*/0);
}
}  // namespace vpux::net
