//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

namespace vpux {

// Get the type of the async value. If the value is not an async value, return its original type.
mlir::Type getAsyncValueType(mlir::Value value);

// Get the DMA type operation from the async execute operation.
VPUIP::DMATypeOpInterface getDmaTypeOp(mlir::async::ExecuteOp execOp);

// Get the executor type from the async execute operation.
VPU::ExecutorKind getExecutorType(mlir::async::ExecuteOp execOp);

}  // namespace vpux
