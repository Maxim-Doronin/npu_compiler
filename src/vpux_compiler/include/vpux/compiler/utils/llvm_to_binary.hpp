//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/utils/logger/logger.hpp>

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {

void translateToLLVMIR(mlir::ModuleOp moduleOp, mlir::SymbolRefAttr swKernelSymbol, vpux::Logger log);
void lowerLLVMToBinary(mlir::ModuleOp moduleOp, mlir::SymbolRefAttr swKernelSymbol);
// Clones the LLVMFunc referenced by swKernelSymbol from srcModuleOp into dstModuleOp
// along with any other referenced LLVMFuncs.
void transitivelyCloneFunctions(mlir::ModuleOp dstModuleOp, mlir::ModuleOp srcModuleOp,
                                mlir::SymbolRefAttr swKernelSymbol);

}  // namespace vpux
