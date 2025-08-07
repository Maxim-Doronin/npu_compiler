//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include "vpux/compiler/dialect/IE/interfaces/convert_quantize_ops_to_nce_ops_strategy.hpp"

namespace vpux::IE {

std::unique_ptr<IConvertQuantizeOpsToNceOpsStrategy> createConvertQuantizeOpsToNceOpsStrategy(
        mlir::func::FuncOp funcOp);

}  // namespace vpux::IE
