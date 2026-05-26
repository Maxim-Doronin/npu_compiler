//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_TYPES
#define VPUX_COMPILER_DIALECT_BYTECODE_TYPES

include "vpux/compiler/dialect/bytecode/dialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

class Bytecode_Type<string name, list<Trait> traits = []>
    : TypeDef<Bytecode_Dialect, name, traits> {
  let mnemonic = name;
}

def Bytecode_RegisterType : Bytecode_Type<"Register"> {
    let summary = "Bytecode register type";
    let description = [{
        This type represents a register in the bytecode dialect.
        A register represents a 64-bit container and is identified by a 16-bit signed number.
    }];
}

#endif
