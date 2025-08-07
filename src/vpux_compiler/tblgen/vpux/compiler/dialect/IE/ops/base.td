//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_BASE
#define VPUX_COMPILER_DIALECT_IE_OPS_BASE

include "vpux/compiler/dialect/IE/dialect.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IE_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            IE_Dialect,
            mnemonic,
            traits
        >;

class IE_LayerOp<string mnemonic, list<Trait> traits = []> :
        IE_Op<
            mnemonic,
            [
                Pure,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<IE_LayerOpInterface>
            ] # traits
        > {
    list<string> elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    bit checkInferredDimsOrder = 0;
    bit checkInferredMemSpace = 0;

    code baseExtraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::areTypesCompatible(lhs, rhs,
                }] # !interleave(elemComparisonModes, "|") # [{,
                static_cast<bool>(}] # checkInferredDimsOrder # [{),
                static_cast<bool>(}] # checkInferredMemSpace # [{)
            );
        }
    }];

    let extraClassDeclaration = baseExtraClassDeclaration;

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

#endif
