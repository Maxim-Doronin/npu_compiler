//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.func_section @function_section {
    bytecode.ext.func @add (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.add.i32 %dst, %lhs, %rhs
        bytecode.add.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @mul (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.mul.i32 %dst, %lhs, %rhs
        bytecode.mul.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @min (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.min.i32 %dst, %lhs, %rhs
        bytecode.min.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
    bytecode.ext.func @max (i64, i64) -> () {
        %dst = bytecode.virtual_general_register
        %lhs = bytecode.virtual_parameter_register 1
        %rhs = bytecode.virtual_parameter_register 2
        bytecode.max.i32 %dst, %lhs, %rhs
        bytecode.max.i64 %dst, %lhs, %rhs
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @add
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.add.i32 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.add.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @mul
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.mul.i32 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.mul.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @min
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.min.i32 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.min.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @max
// CHECK:         [[DST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[LHS:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         [[RHS:%.+]] = bytecode.virtual_parameter_register 2
// CHECK:         bytecode.max.i32 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.max.i64 [[DST]], [[LHS]], [[RHS]]
// CHECK:         bytecode.ret
