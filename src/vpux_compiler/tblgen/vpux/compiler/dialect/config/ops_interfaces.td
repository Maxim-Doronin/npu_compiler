//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_CONFIG_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// DefinedInArch op trait
//

class DefinedInArch<string arch>
    : ParamNativeOpTrait<"DefinedInArch", arch>  {
    let cppNamespace = "vpux::config";
}

//
// LimitedToArch op trait
//

class LimitedToArch<list<string> archs>
    : ParamNativeOpTrait<"LimitedToArch", !interleave(archs, ", ")>  {
    let cppNamespace = "vpux::config";
}

#endif
