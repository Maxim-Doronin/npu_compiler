//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

bool isConstOperandOp(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
