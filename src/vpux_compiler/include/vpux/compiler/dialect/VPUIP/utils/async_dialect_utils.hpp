//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"

#include <mlir/Dialect/Async/IR/Async.h>

namespace vpux::VPUIP {

// Get the executor type from the async execute operation.
config::ExecutorKind getExecutorType(mlir::async::ExecuteOp execOp);

// Get the DMA type operation from the async execute operation.
VPUIP::DMATypeOpInterface getDmaTypeOp(mlir::async::ExecuteOp execOp);

}  // namespace vpux::VPUIP
