//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_BYTECODE_ATTRIBUTES

include "vpux/compiler/dialect/bytecode/dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//
// Base classes
//

class Bytecode_Attr<string cppName, string mnemonicStr, list<Trait> traits = []> :
        AttrDef<Bytecode_Dialect, cppName, traits> {
    let mnemonic = mnemonicStr;
}

//
// Float format enum
//

def Bytecode_FloatFormat_IEEE    : I64EnumAttrCase<"IEEE",    0x00>;
def Bytecode_FloatFormat_BFloat  : I64EnumAttrCase<"BFloat",  0x01>;
def Bytecode_FloatFormat_TFloat  : I64EnumAttrCase<"TFloat",  0x02>;
def Bytecode_FloatFormat_E4M3    : I64EnumAttrCase<"E4M3",    0x03>;
def Bytecode_FloatFormat_E5M2    : I64EnumAttrCase<"E5M2",    0x04>;
def Bytecode_FloatFormat_E2M1    : I64EnumAttrCase<"E2M1",    0x05>;

def Bytecode_FloatFormat : I64EnumAttr<
    "FloatFormat",
    "Bytecode floating-point format",
    [Bytecode_FloatFormat_IEEE, Bytecode_FloatFormat_BFloat,
     Bytecode_FloatFormat_TFloat, Bytecode_FloatFormat_E4M3,
     Bytecode_FloatFormat_E5M2, Bytecode_FloatFormat_E2M1]> {
    let cppNamespace = "vpux::bytecode";
}

//
// Bytecode type attributes
//

def Bytecode_IntegerTypeAttr : Bytecode_Attr<"IntegerType", "integer_type"> {
    let summary = "Bytecode integer type attribute";
    let description = [{
        Represents an integer type in the bytecode type section.
        The width specifies the number of bits.

        Example:
        ```mlir
        bytecode.type @i64 #bytecode.integer_type<width = 64>
        ```
    }];
    let parameters = (ins
        "int64_t":$width
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

def Bytecode_FloatTypeAttr : Bytecode_Attr<"FloatType", "float_type"> {
    let summary = "Bytecode floating-point type attribute";
    let description = [{
        Represents a floating-point type in the bytecode type section.
        The width specifies the number of bits, and the format specifies
        the floating-point encoding.

        Example:
        ```mlir
        bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE>
        ```
    }];
    let parameters = (ins
        "int64_t":$width,
        EnumParameter<Bytecode_FloatFormat>:$format
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

def Bytecode_OpaqueTypeAttr : Bytecode_Attr<"OpaqueType", "opaque_type"> {
    let summary = "Bytecode opaque type attribute";
    let description = [{
        Represents an opaque type in the bytecode type section.
        The width specifies the number of bits.

        Example:
        ```mlir
        bytecode.type @opaque8 #bytecode.opaque_type<width = 8>
        ```
    }];
    let parameters = (ins
        "int64_t":$width
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

def Bytecode_BufferTypeAttr : Bytecode_Attr<"BufferType", "buffer_type"> {
    let summary = "Bytecode buffer type attribute";
    let description = [{
        Represents a buffer type in the bytecode type section.
        The element type references another type in the type section via a symbol.
        Shape and strides use -1 to represent dynamic dimensions.

        Example:
        ```mlir
        bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 4,
            shape = [1, 16, 32, 32], strides = [16384, 1024, 32, 1]>
        ```
    }];
    let parameters = (ins
        "mlir::FlatSymbolRefAttr":$element_type,
        "int64_t":$rank,
        "::mlir::DenseI64ArrayAttr":$shape,
        "::mlir::DenseI64ArrayAttr":$strides
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

def Bytecode_FunctionTypeAttr : Bytecode_Attr<"FunctionType", "function_type"> {
    let summary = "Bytecode function type attribute";
    let description = [{
        Represents a function type in the bytecode type section.
        Arguments and results reference other types in the type section via symbols.

        Example:
        ```mlir
        bytecode.type @fn_type #bytecode.function_type<
            arguments = [@buf_type], results = [@i64]>
        ```
    }];
    let parameters = (ins
        "::mlir::ArrayAttr":$arguments,
        "::mlir::ArrayAttr":$results
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

#endif
