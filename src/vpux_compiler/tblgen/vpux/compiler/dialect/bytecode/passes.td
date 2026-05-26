//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_PASSES
#define VPUX_COMPILER_DIALECT_BYTECODE_PASSES

include "mlir/Pass/PassBase.td"

//
// ConvertIntermediateBytecodeOps
//

def ConvertIntermediateBytecodeOps : PassBase<"convert-intermediate-bytecode-ops", "vpux::ModulePass"> {
    let summary = "Converts intermediate bytecode operations to bytecode instructions";

    let description = [{
        The pass converts intermediate bytecode operations to bytecode instructions.
        These operations use external data (e.g. strings) directly. This data needs to be moved to the
        appropriate sections (e.g. string section) before the final bytecode instructions can be created.
        This pass performs this transformation. The newly created instructions use symbols to refer to the external data.

        Examples of intermediate operations are `bytecode.ext.assert`, which can be replaced by
        `bytecode.assert` after moving the message to the string section.
    }];

    let constructor = "vpux::bytecode::createConvertIntermediateBytecodeOpsPass()";

    let dependentDialects = [
        "vpux::bytecode::BytecodeDialect"
    ];
}

//
// SerializeKernelsToBytecode
//

def SerializeKernelsToBytecode : PassBase<"serialize-kernels-to-bytecode", "vpux::ModulePass"> {
    let summary = "Convert HostExec::BinaryOp operations into bytecode kernel section";

    let description = [{
        This pass converts HostExec::BinaryOp operations (containing serialized ELF binary data)
        into bytecode kernel_section/kernel operations.

        It walks HostExec::BinaryOp instances, extracts the binary data from the contained
        HostExec::BinaryDataOp, and creates bytecode.kernel ops in a bytecode.kernel_section.
        The BinaryOp is then erased and replaced with a stub mlir::ModuleOp containing a
        function declaration, to preserve symbol resolution for Core::NestedCallOp references.
    }];

    let constructor = "vpux::bytecode::createSerializeKernelsToBytecodePass()";

    let dependentDialects = [
        "vpux::bytecode::BytecodeDialect",
        "vpux::HostExec::HostExecDialect"
    ];
}

#endif
