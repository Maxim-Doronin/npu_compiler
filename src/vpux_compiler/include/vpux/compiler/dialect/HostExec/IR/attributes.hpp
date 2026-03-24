//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/attr_interfaces.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/HostExec/attributes.hpp.inc>

namespace vpux {
namespace HostExec {

/// @brief The attribute @HostCompileInferenceExec is used to specify
/// a FuncOp in a module, which is about to be a target for
/// converting MLIR operations to LLVM UMD (User Mode Driver) runtime API calls.
/// This conversion happens in the ConvertToLLVMUMDCallsPass.
void setHostCompileInferenceExecFuncAttribute(mlir::func::FuncOp func);
bool isHostCompileInferenceExecFunc(mlir::func::FuncOp func);
}  // namespace HostExec
}  // namespace vpux
