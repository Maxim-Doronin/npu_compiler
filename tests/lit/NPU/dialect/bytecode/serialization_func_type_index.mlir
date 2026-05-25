//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-bytecode %s -o %t
// RUN: bytecode_interpreter --path %t --mode print-full | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// Test that each bytecode.func resolves its function_type_ref to the correct
// positional index in the type section.
//
// Type section layout:
//   index 0: @i64_type  (i64)
//   index 1: @f32_type  (f32)
//   index 2: @fn_type_a (function_type with i64 args)
//   index 3: @fn_type_b (function_type with f32 args)
//
// @add  references @fn_type_a -> expected function type index: 2
// @scale references @fn_type_b -> expected function type index: 3

module {
bytecode.func_section @function_section {
    bytecode.func @add @fn_type_a {
        %dst = bytecode.general_register 0
        %lhs = bytecode.general_register 1
        %rhs = bytecode.general_register 2
        bytecode.add.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.func @scale @fn_type_b {
        %dst = bytecode.general_register 0
        %src = bytecode.general_register 1
        bytecode.set %dst, %src
        bytecode.ret
    }
}
bytecode.type_section @type_section {
    bytecode.type @i64_type i64
    bytecode.type @f32_type f32
    bytecode.type @fn_type_a #bytecode.function_type<arguments = [@i64_type, @i64_type], results = [@i64_type]>
    bytecode.type @fn_type_b #bytecode.function_type<arguments = [@f32_type], results = [@f32_type]>
}
}

// CHECK:        Number of functions: 2, entrypoint function index: 0
// CHECK-NEXT:     Name index: 0, function type index: 2, num general registers: 3
// CHECK-NEXT:     Name index: 0, function type index: 3, num general registers: 2
