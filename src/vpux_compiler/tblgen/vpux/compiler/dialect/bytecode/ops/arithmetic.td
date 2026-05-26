//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_ARITHMETIC
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_ARITHMETIC

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"

def Bytecode_AddI32Op : Bytecode_ThreeOperandsInstructionBase<"add.i32", [], "1.0.0", "vpux::bytecode::OpCode::ADD_I32"> {
    let summary = "Add two 32-bit integers";

    let opDescription = [{
        This operation adds two 32-bit integers (`lhs`, `rhs`) and produces a 32-bit integer result (`dst`).

        The integers are expected to use two's complement representation, making this instruction appropriate for both signed and unsigned integer addition.
        TODO: overflow behavior

        Examples:
        ```mlir
        bytecode.add.i32 %dst, %lhs, %rhs  // register mode
        bytecode.add.i32 %dst, %lhs, 10    // immediate mode
        ```
    }];
}

def Bytecode_AddI64Op : Bytecode_ThreeOperandsInstructionBase<"add.i64", [], "1.0.0", "vpux::bytecode::OpCode::ADD_I64"> {
    let summary = "Add two 64-bit integers";

    let opDescription = [{
        This operation adds two 64-bit integers (`lhs`, `rhs`) and produces a 64-bit integer result (`dst`).

        The integers are expected to use two's complement representation, making this instruction appropriate for both signed and unsigned integer addition.
        TODO: overflow behavior

        Examples:
        ```mlir
        bytecode.add.i64 %dst, %lhs, %rhs  // register mode
        bytecode.add.i64 %dst, %lhs, 10    // immediate mode
        ```
    }];
}

def Bytecode_MulI32Op : Bytecode_ThreeOperandsInstructionBase<"mul.i32", [], "1.0.0", "vpux::bytecode::OpCode::MUL_I32"> {
    let summary = "Multiply two 32-bit integers";

    let opDescription = [{
        This operation multiplies two 32-bit integers (`lhs`, `rhs`) and produces a 32-bit integer result (`dst`).

        The integers are expected to use two's complement representation, making this instruction appropriate for both signed and unsigned integer multiplication.
        TODO: overflow behavior

        Examples:
        ```mlir
        bytecode.mul.i32 %dst, %lhs, %rhs  // register mode
        bytecode.mul.i32 %dst, %lhs, 10    // immediate mode
        ```
    }];
}

def Bytecode_MulI64Op : Bytecode_ThreeOperandsInstructionBase<"mul.i64", [], "1.0.0", "vpux::bytecode::OpCode::MUL_I64"> {
    let summary = "Multiply two 64-bit integers";

    let opDescription = [{
        This operation multiplies two 64-bit integers (`lhs`, `rhs`) and produces a 64-bit integer result (`dst`).

        The integers are expected to use two's complement representation, making this instruction appropriate for both signed and unsigned integer multiplication.
        TODO: overflow behavior

        Examples:
        ```mlir
        bytecode.mul.i64 %dst, %lhs, %rhs  // register mode
        bytecode.mul.i64 %dst, %lhs, 10    // immediate mode
        ```
    }];
}

def Bytecode_MinI32Op : Bytecode_ThreeOperandsInstructionBase<"min.i32", [], "1.0.0", "vpux::bytecode::OpCode::MIN_I32"> {
    let summary = "Compute the signed minimum of two 32-bit integers";

    let opDescription = [{
        This operation computes the signed minimum of two 32-bit integers (`lhs`, `rhs`) and stores the result in `dst`.

        Both operands are interpreted as two's complement signed 32-bit integers.

        Examples:
        ```mlir
        bytecode.min.i32 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MinI64Op : Bytecode_ThreeOperandsInstructionBase<"min.i64", [], "1.0.0", "vpux::bytecode::OpCode::MIN_I64"> {
    let summary = "Compute the signed minimum of two 64-bit integers";

    let opDescription = [{
        This operation computes the signed minimum of two 64-bit integers (`lhs`, `rhs`) and stores the result in `dst`.

        Both operands are interpreted as two's complement signed 64-bit integers.

        Examples:
        ```mlir
        bytecode.min.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MaxI32Op : Bytecode_ThreeOperandsInstructionBase<"max.i32", [], "1.0.0", "vpux::bytecode::OpCode::MAX_I32"> {
    let summary = "Compute the signed maximum of two 32-bit integers";

    let opDescription = [{
        This operation computes the signed maximum of two 32-bit integers (`lhs`, `rhs`) and stores the result in `dst`.

        Both operands are interpreted as two's complement signed 32-bit integers.

        Examples:
        ```mlir
        bytecode.max.i32 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MaxI64Op : Bytecode_ThreeOperandsInstructionBase<"max.i64", [], "1.0.0", "vpux::bytecode::OpCode::MAX_I64"> {
    let summary = "Compute the signed maximum of two 64-bit integers";

    let opDescription = [{
        This operation computes the signed maximum of two 64-bit integers (`lhs`, `rhs`) and stores the result in `dst`.

        Both operands are interpreted as two's complement signed 64-bit integers.

        Examples:
        ```mlir
        bytecode.max.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

#endif
