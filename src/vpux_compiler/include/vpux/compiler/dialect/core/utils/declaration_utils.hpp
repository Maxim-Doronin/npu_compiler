//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux {

void moveDeclarationsToTop(mlir::func::FuncOp& netFunc);

}  // namespace vpux
