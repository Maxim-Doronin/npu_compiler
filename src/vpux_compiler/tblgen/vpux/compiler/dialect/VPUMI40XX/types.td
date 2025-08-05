//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI40XX_TYPES
#define VPUX_COMPILER_DIALECT_VPUMI40XX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "vpux/compiler/dialect/VPUMI40XX/dialect.td"

class VPUMI40XX_Type<string name, list<Trait> traits = [],
                          string baseCppClass = "::mlir::Type">
        : TypeDef<VPUMI40XX_Dialect, name, traits, baseCppClass> {
    let mnemonic = name;
}

#endif
