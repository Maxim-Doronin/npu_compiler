//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_HOSTEXEC_OPS
#define VPUX_COMPILER_DIALECT_HOSTEXEC_OPS

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"

include "vpux/compiler/dialect/HostExec/dialect.td"
include "vpux/compiler/dialect/HostExec/attributes.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class HostExec_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            HostExec_Dialect,
            mnemonic,
            traits
        >;

//
// HostExec_BinaryOp
//

def HostExec_BinaryOp :
        HostExec_Op<
            "Binary",
            [
                SingleBlock,
                IsolatedFromAbove,
                NoTerminator,
                // must be directly under ModuleOp
                HasParent<"mlir::ModuleOp">,
                // disallow any other ops other than const.Rodata in body
                Core_HasChild<["vpux::HostExec::BinaryDataOp", "mlir::func::FuncOp"]>,
                SymbolTable,
                Symbol
            ]
        > {
    let summary = "Container op that holds serialized ELF binary data and function call declarations in ELF binary";

    let description = [{
        This operation is a container that holds serialized ELF binary data.
        Along with binary data, it holds external function declarations that represent
        the entry point functions in serialized ELF compute function.

        Example:

        ``` MLIR
        HostExec.Binary (symbolTable) {
            HostExec.BinaryDataOp @kernel <object = "\7FELF\02\01\00\00\00\{{.*}}">
            func.func private @kernel (...) -> (...)
        }

        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $sym_name $body attr-dict
    }];

    // we need a custom builder to add a single empty block
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::StringRef":$sym_name), [{
            $_state.getOrAddProperties<Properties>().sym_name = $_builder.getStringAttr(sym_name);
            auto region = $_state.addRegion();
            region->push_back(new mlir::Block());
        }]>
    ];
}

//
// HostExec_BinaryDataOp
//

def HostExec_BinaryDataOp :
        HostExec_Op<
            "BinaryData",
            [
                Symbol,
                HasParent<"vpux::HostExec::BinaryOp">
            ]
            > {
  let summary = "An Op for storing serialized HostExec binary objects.";

  let description = [{
    An Object attribute that holds binary string of serialized ELF data.
    This binary object is lowered down to LLVM IR in host compilation pipeline
  }];

  let arguments = (ins
        SymbolNameAttr:$sym_name,
        HostExec_ObjectAttr:$object
    );

  let builders = [
    OpBuilder<(ins "mlir::StringRef":$name,
                   "mlir::Attribute":$object)>,
  ];

  let assemblyFormat = [{
    $sym_name attr-dict $object
  }];
}


#endif
