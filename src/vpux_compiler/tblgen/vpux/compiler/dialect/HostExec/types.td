//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_HOSTEXEC_TYPES
#define VPUX_COMPILER_DIALECT_HOSTEXEC_TYPES

include "vpux/compiler/dialect/HostExec/dialect.td"
include "vpux/compiler/dialect/HostExec/attributes.td"
include "vpux/compiler/dialect/core/type_interfaces.td"

include "mlir/IR/BuiltinTypeInterfaces.td"

// Base class for HostExec dialect types.
class HostExec_Type<string name, list<Trait> traits = []>
    : TypeDef<HostExec_Dialect, name, traits> {
  let mnemonic = name;
}

#endif
