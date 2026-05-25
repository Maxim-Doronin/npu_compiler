//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --split-input-file --vpu-arch=%arch% --export-bytecode %s -o %t
// RUN: bytecode_interpreter --path %t --mode print-full | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module {
bytecode.func_section @function_section {
    bytecode.func @add @fn_type {
        %dst = bytecode.general_register 0
        %lhs = bytecode.general_register 1
        %rhs = bytecode.general_register 2

        bytecode.set_imm %lhs, 10
        bytecode.set %rhs, %lhs
        bytecode.add.i32 %dst, %lhs, %rhs

        bytecode.set_imm %rhs, -20
        bytecode.add.i64 %dst, %dst, %rhs

        bytecode.assert %dst, @another_string
        bytecode.ret
    }
    bytecode.func @mul @fn_type {
        %dst = bytecode.general_register 0
        %lhs = bytecode.general_register 1
        %rhs = bytecode.general_register 2
        bytecode.mul.i32 %dst, %lhs, %rhs
        bytecode.mul.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.func @min @fn_type {
        %dst = bytecode.general_register 0
        %lhs = bytecode.general_register 1
        %rhs = bytecode.general_register 2
        bytecode.min.i32 %dst, %lhs, %rhs
        bytecode.min.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.func @max @fn_type {
        %dst = bytecode.general_register 0
        %lhs = bytecode.general_register 1
        %rhs = bytecode.general_register 2
        bytecode.max.i32 %dst, %lhs, %rhs
        bytecode.max.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
}
bytecode.constant_section @constant_section {
    bytecode.constant @my_constant dense<[42, 100, 50]> : tensor<3xi64>
    bytecode.constant @another_constant dense<[3.14]> : tensor<1xf32>
}
bytecode.string_section @string_section {
    bytecode.string @my_string "Example of a string"
    bytecode.string @another_string "Another example"
}
bytecode.type_section @type_section {
    bytecode.type @i64_type i64
    bytecode.type @f32_type f32
    bytecode.type @fn_type #bytecode.function_type<arguments = [@i64_type, @i64_type], results = []>
}
}

// CHECK:  Magic Number: 4E 50 55 42 79 74 65 00
// CHECK:  Version: 1.0.0
// CHECK:  Section Header Table:
// CHECK:    Number of sections: 4
// CHECK:      Section type: Function, name index: 0, offset: 434, size: 108
// CHECK:        Number of functions: 4, entrypoint function index: 0
// CHECK:          Name index: 0, function type index: 2, num general registers: 3, body offset: 0, body size: 54
// CHECK:          Name index: 0, function type index: 2, num general registers: 3, body offset: 54, body size: 18
// CHECK:          Name index: 0, function type index: 2, num general registers: 3, body offset: 72, body size: 18
// CHECK:          Name index: 0, function type index: 2, num general registers: 3, body offset: 90, body size: 18
// CHECK:      Section type: Constant, name index: 0, offset: 542, size: 28
// CHECK:        Number of entries: 2
// CHECK:          Entry 0 offset: 0, size: 24
// CHECK:          Entry 1 offset: 24, size: 4
// CHECK:      Section type: String, name index: 0, offset: 570, size: 36
// CHECK:        Number of entries: 2
// CHECK:          Entry 0 offset: 0, size: 20
// CHECK:          Entry 1 offset: 20, size: 16
// CHECK:      Section type: Type, name index: 0, offset: 606, size: 26
// CHECK:        Number of entries: 3
// CHECK:          Entry 0 offset: 0, size: 2
// CHECK:          Entry 1 offset: 2, size: 3
// CHECK:          Entry 2 offset: 5, size: 21
// CHECK:    Function section 0
// CHECK:      Function name: 0
// CHECK:        set.imm 1, 10
// CHECK:        set 2, 1
// CHECK:        add.i32 0, 1, 2
// CHECK:        set.imm 2, -20
// CHECK:        add.i64 0, 0, 2
// CHECK:        assert 0, 1
// CHECK:        ret
// CHECK:      Function name: 0
// CHECK:        mul.i32 0, 1, 2
// CHECK:        mul.i64 0, 1, 2
// CHECK:        ret
// CHECK:      Function name: 0
// CHECK:        min.i32 0, 1, 2
// CHECK:        min.i64 0, 1, 2
// CHECK:        ret
// CHECK:      Function name: 0
// CHECK:        max.i32 0, 1, 2
// CHECK:        max.i64 0, 1, 2
// CHECK:        ret
// CHECK:    Constant section 0
// CHECK:      Constant 0: 0x2A0000000000000064000000000000003200000000000000
// CHECK:      Constant 1: 0xC3F54840
// CHECK:    String section 0
// CHECK:      String 0: Example of a string\0
// CHECK:      String 1: Another example\0
// CHECK:    Type section 0
// CHECK:      Type 0:
// CHECK:      Type 1:
// CHECK:      Type 2:
