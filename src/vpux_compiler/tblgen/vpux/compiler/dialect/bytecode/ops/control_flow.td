//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_CONTROL_FLOW
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_CONTROL_FLOW

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"

def Bytecode_RetOp : Bytecode_ZeroOperandsInstructionBase<"ret", [Terminator], "1.0.0", "vpux::bytecode::OpCode::RET"> {
    let summary = "Return operation";

    let opDescription = [{
        This operation represents a return operation in the bytecode format, and it is used to terminate the execution of a function.
        It does not return any value.

        Example:
        ```mlir
        bytecode.ret
        ```
    }];
}

def Bytecode_AssertOp : Bytecode_InstructionOp<"assert", [], "1.0.0", "vpux::bytecode::OpCode::ASSERT"> {
    let summary = "Assert operation";

    let description = [{
        This operation represents an assert operation in the bytecode format, and it is used to terminate the execution of a function if a condition is not met.
        The message operand corresponds with the index of a string within the string section.

        Example:
        ```mlir
        bytecode.assert %condition, %message  // Asserts that the condition is true, and if not, terminates the function execution with the provided message
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$condition,
        SymbolNameAttr:$msg_sym
    );

    let assemblyFormat = [{
        $condition `,` $msg_sym attr-dict
    }];
}

#endif
