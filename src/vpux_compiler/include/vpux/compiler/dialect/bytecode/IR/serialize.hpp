//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

namespace vpux::bytecode {

mlir::LogicalResult serializeTo(mlir::ModuleOp moduleOp, llvm::raw_ostream& os);

}  // namespace vpux::bytecode
