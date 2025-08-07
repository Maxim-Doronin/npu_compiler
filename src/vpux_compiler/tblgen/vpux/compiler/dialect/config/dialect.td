//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG
#define VPUX_COMPILER_DIALECT_CONFIG

include "mlir/IR/OpBase.td"

def Config_Dialect : Dialect {
    let summary = "Config Dialect";

    let description = [{
        The `config` dialect contains items related to the configuration of the compilation.
        For example, operations which describe pipeline options or operations which describe
        the amount of NPU resources that are allocated for the current compilation.
    }];

    let name = "config";

    let cppNamespace = "vpux::config";

    let extraClassDeclaration = [{
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
}

#endif
