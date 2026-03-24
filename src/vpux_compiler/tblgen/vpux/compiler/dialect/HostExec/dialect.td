//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_HOSTEXEC
#define VPUX_COMPILER_DIALECT_HOSTEXEC

include "mlir/IR/OpBase.td"

def HostExec_Dialect : Dialect {
    let summary = "Host Execution Dialect";

    let description = [{
The **HostExec Dialect** represents Host execution IR in terms of MLIR framework.

It allows to work with  inside MLIR framework:

* Perform additional low level transformations
** Serialize size ELF MLIR by lowering it to a BinaryOp
** Lower AsyncOps to L0 wrapper function calls

It handles such NPU-specifics as:
* Supported operation set.

[./HostExec/_ops_interfaces.md]
    }];

    let name = "HostExec";

    let cppNamespace = "vpux::HostExec";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::Core::CoreDialect"
    ];

    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;

    let usePropertiesForAttributes = 1;

    let extraClassDeclaration = [{
        void registerAttributes();
    }];
}

#endif
