//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::WorkloadCastOp::getViewSource() {
    return getInput();
}
