//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_CORE_OPS
#define VPUX_COMPILER_DIALECT_CORE_OPS

include "vpux/compiler/dialect/core/dialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//
// Base classes
//

class Core_Op<string mnemonic, list<Trait> traits = []> :
    Op<Core_Dialect, mnemonic, traits>;

//
// NestedCallOp
//

def Core_NestedCallOp : Core_Op<"NestedCall", [
      CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "An operation that is similar to `func.call` but supports nested symbols";

    let description = [{
        This operation fulfills the same purpose as `func.call`, except for that it *only* supports nested symbols. The verifier checks if the symbol points to a `func.func` operation with a matching type signature.

        Example:

        ```MLIR
        module @subModule {
            func.func @foo() -> () { ... }
        }

        func.func @bar() {
            Core.NestedCall @subModule::@foo()
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$arg_operands, SymbolRefAttr:$callee);

    let results = (outs Variadic<AnyType>);

    // implementation of CallOpInterface
    let extraClassDeclaration = [{
        /// Return the callee of this operation.
        mlir::CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
            setCalleeAttr(callee.get<mlir::SymbolRefAttr>());
        }
    }];

    let builders = [
        OpBuilder<(ins "mlir::SymbolRefAttr":$callee, "mlir::TypeRange":$results, CArg<"mlir::ValueRange", "{}">:$operands)>
    ];

    let assemblyFormat = [{
        $callee `(` $arg_operands `)` attr-dict `:` functional-type($arg_operands, results)
    }];
}

#endif
