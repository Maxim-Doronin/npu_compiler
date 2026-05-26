//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK: bytecode.type @i64 #bytecode.integer_type<width = 64>
// CHECK: bytecode.type @i32 #bytecode.integer_type<width = 32>
// CHECK: bytecode.type @i8 #bytecode.integer_type<width = 8>
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64>
    bytecode.type @i32 #bytecode.integer_type<width = 32>
    bytecode.type @i8 #bytecode.integer_type<width = 8>
}

// -----

// CHECK: bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE>
// CHECK: bytecode.type @f64 #bytecode.float_type<width = 64, format = IEEE>
// CHECK: bytecode.type @bf16 #bytecode.float_type<width = 16, format = BFloat>
bytecode.type_section @type_section {
    bytecode.type @f32 #bytecode.float_type<width = 32, format = IEEE>
    bytecode.type @f64 #bytecode.float_type<width = 64, format = IEEE>
    bytecode.type @bf16 #bytecode.float_type<width = 16, format = BFloat>
}

// -----

// CHECK: bytecode.type @opaque8 #bytecode.opaque_type<width = 8>
bytecode.type_section @type_section {
    bytecode.type @opaque8 #bytecode.opaque_type<width = 8>
}

// -----

// CHECK: bytecode.type @i64 #bytecode.integer_type<width = 64>
// CHECK: bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 4, shape = [1, 16, 32, 32], strides = [16384, 1024, 32, 1]>
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64>
    bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 4, shape = [1, 16, 32, 32], strides = [16384, 1024, 32, 1]>
}

// -----

// CHECK: bytecode.type @i64 #bytecode.integer_type<width = 64>
// CHECK: bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 2, shape = [4, 8], strides = [8, 1]>
// CHECK: bytecode.type @fn_type #bytecode.function_type<arguments = [@buf], results = [@i64]>
bytecode.type_section @type_section {
    bytecode.type @i64 #bytecode.integer_type<width = 64>
    bytecode.type @buf #bytecode.buffer_type<element_type = @i64, rank = 2, shape = [4, 8], strides = [8, 1]>
    bytecode.type @fn_type #bytecode.function_type<arguments = [@buf], results = [@i64]>
}

// -----

// Test raw MLIR types via TypeAttr
// CHECK: bytecode.type @a i64
// CHECK: bytecode.type @b f32
// CHECK: bytecode.type @c bf16
bytecode.type_section @type_section {
    bytecode.type @a i64
    bytecode.type @b f32
    bytecode.type @c bf16
}
