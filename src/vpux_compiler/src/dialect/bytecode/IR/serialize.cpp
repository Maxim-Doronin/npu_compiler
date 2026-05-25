//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/serialize.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;

mlir::LogicalResult vpux::bytecode::serializeTo(mlir::ModuleOp moduleOp, llvm::raw_ostream& os) {
    BytecodeWriter writer(moduleOp);
    writer.appendFileHeader();
    writer.appendSections();
    writer.writeTo(os);
    return mlir::success();
}
