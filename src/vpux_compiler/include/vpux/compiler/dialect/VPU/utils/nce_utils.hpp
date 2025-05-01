//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux {
namespace VPU {
bool isDepthwiseOp(mlir::Operation* op);
}  // namespace VPU
}  // namespace vpux
