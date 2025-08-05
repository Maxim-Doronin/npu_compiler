//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG_OPS
#define VPUX_COMPILER_DIALECT_CONFIG_OPS

include "vpux/compiler/dialect/config/dialect.td"
include "vpux/compiler/dialect/core/attributes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class Config_Op<string mnemonic, list<Trait> traits = []> :
    Op<Config_Dialect, mnemonic, traits>;

def Config_PipelineOptionsOp :
        Config_Op<
            "PipelineOptions",
            [
                SymbolTable,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Parallel to config options used in the compiler to be exposed globally";

    let description = [{
        Options can include another PipelineOptionsOp or OptionsOp
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$options
    );

    let assemblyFormat = [{
        attr-dict
        $sym_name
        custom<OptionalBlockRegion>($options)
    }];
}

def Config_OptionOp :
        Config_Op<
            "Option",
            [
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Option to be added to the model/IR";

    let description = [{
        Concrete option values that will be added to IR, must have a sym_name for lookup
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        AnyAttrOf<[IntAttr, F32Attr, BoolAttr]>:$optionValue
    );

    let assemblyFormat = [{
        $sym_name `:` $optionValue
        attr-dict
    }];
}

#endif
