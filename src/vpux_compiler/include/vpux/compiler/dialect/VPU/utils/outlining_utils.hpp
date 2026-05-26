//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/function_outlining_splitter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

bool isConstOperandOp(mlir::Operation* op);

bool isConstantLikeOp(mlir::Operation* op);

// Optimize outlined functions by removing unused constant outputs.
// During outlining, constants (Const::DeclareOp) and constant-like operations (SE tables, DP tables, ZP tables)
// are duplicated into each outlined function body. By default, the outlining process conservatively includes
// these as function outputs. However, many of them are only needed internally within the outlined function
// (e.g., DP tables and ZP tables are consumed by NCE operations inside the function) and have no consumers
// at the call site in the main function. This function removes such unused constant outputs from:
// - The outlined function's signature and return operation
// - The corresponding call operation in the main function
// This reduces unnecessary data transfers and simplifies the IR without affecting correctness, as these
// operations remain duplicated inside each outlined function where they are actually needed.
void removeUnusedConstantOutputs(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                                 ArrayRef<OutliningInstance> outliningInstances, const Logger& log);

}  // namespace VPU
}  // namespace vpux
