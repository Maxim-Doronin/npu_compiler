//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_CONFIG_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/config/dialect.td"

//
// Base classes
//

class Config_Attr<string name, list<Trait> traits = []> : AttrDef<Config_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

class Config_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::config";
    let genSpecializedAttr = 0;
}

class Config_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<Config_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

//
// CompilationMode
//

def Config_CompilationMode  :
        Config_I64EnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                I64EnumAttrCase<"ReferenceSW",  0>,
                I64EnumAttrCase<"DefaultHW",    1>,
                I64EnumAttrCase<"ShaveCodeGen", 2>,
                I64EnumAttrCase<"HostCompile",  3>,
                I64EnumAttrCase<"WSMonolithic", 4>,
                I64EnumAttrCase<"WSInit",       5>
            ]
        > {
}

def Config_CompilationModeAttr : Config_EnumAttr<Config_CompilationMode, "compilation_mode">;

#endif
