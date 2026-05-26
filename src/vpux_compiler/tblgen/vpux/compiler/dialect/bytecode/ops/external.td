//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_EXTERNAL
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_EXTERNAL

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"

def Bytecode_ExtAssertOp : Bytecode_Op<"ext.assert", []> {
    let summary = "External assert operation";

    let description = [{
        This operation represents an intermediary stage of an assert instruction, which stores the string message as an attribute.
        This message is external data, which will be moved to the string section when lowering to the final assert instruction operation.

        Example:
        ```mlir
        bytecode.ext.assert %condition, "Assert message"
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$condition,
        StrAttr:$message
    );

    let assemblyFormat = [{
        $condition `,` $message attr-dict
    }];
}

def Bytecode_ExtFuncOp : Bytecode_Op<"ext.func", [HasParent<"vpux::bytecode::FuncSectionOp">, IsolatedFromAbove]> {
    let summary = "External function operation";

    let description = [{
        This operation represents an intermediary stage of a function, which stores the function type as an attribute.
        This type is external data, which will be decomposed into bytecode type attributes and moved to the type section
        when lowering to the final function operation.

        Example:
        ```mlir
        bytecode.ext.func @my_function (i64, i64) -> (i64) {
            // Bytecode instructions for the function body
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let assemblyFormat = [{
        $sym_name $function_type attr-dict-with-keyword
        $body
    }];
}

#endif
