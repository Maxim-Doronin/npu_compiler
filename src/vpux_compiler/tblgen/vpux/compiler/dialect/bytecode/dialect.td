//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE
#define VPUX_COMPILER_DIALECT_BYTECODE

include "mlir/IR/OpBase.td"

def Bytecode_Dialect : Dialect {
    let summary = "Bytecode Dialect";

    let name = "bytecode";

    let cppNamespace = "vpux::bytecode";

    let dependentDialects = [
        "vpux::Core::CoreDialect"
    ];

    let extraClassDeclaration = [{
        void registerAttributes();
        void registerTypes();
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];

    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

#endif
