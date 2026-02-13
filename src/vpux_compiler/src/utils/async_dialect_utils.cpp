
//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/async_dialect_utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>

using namespace vpux;

mlir::Type vpux::getAsyncValueType(mlir::Value value) {
    auto type = value.getType();
    if (const auto asyncType = mlir::dyn_cast<mlir::async::ValueType>(type)) {
        type = asyncType.getValueType();
    }
    return type;
}
