//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"

#include <mlir/IR/Operation.h>

namespace vpux::VPU {

vpux::DimsOrder getTargetOrder(mlir::Operation* permuteOp);

bool isSupportedPermutation(mlir::Operation* nceOp, mlir::Operation* permuteOp);

}  // namespace vpux::VPU
