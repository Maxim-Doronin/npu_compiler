//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE {

/*
   Find right class to get strategies for particular platform
*/
std::unique_ptr<IConversionPassStrategy> createConvertToPalletizationLUTStrategy(mlir::func::FuncOp funcOp);

}  // namespace vpux::IE
