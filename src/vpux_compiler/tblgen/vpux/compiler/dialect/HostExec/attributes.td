//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_HOSTEXEC_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_HOSTEXEC_ATTRIBUTES

include "vpux/compiler/dialect/core/attr_interfaces.td"
include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/HostExec/dialect.td"

//
// Base classes
//

class HostExec_Attr<string name, list<Trait> traits = []> :
        AttrDef<HostExec_Dialect, name, traits> {
    let mnemonic = name # "Attr";
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// HostExecObjectAttr
//

def HostExec_ObjectAttr : HostExec_Attr<"Object"> {
    let description = [{
        HostExec object holds the hex string of serialized NPU compute functions in the ELF dialect.
    }];
    let parameters = (ins
        "mlir::StringAttr":$object
    );
}

#endif
