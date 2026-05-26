//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-intermediate-bytecode-ops %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Test ext.assert -> assert conversion and ext.func -> func conversion
module {

bytecode.func_section @func_section {
  bytecode.ext.func @fn () -> () {
    %0 = bytecode.virtual_general_register
    bytecode.set_imm %0, 0
    bytecode.ext.assert %0, "Assertion failed"
    bytecode.ret
  }
}

// CHECK:  bytecode.string_section @string_section {
// CHECK:    bytecode.string @assert_msg_0 "Assertion failed"
// CHECK:  }
// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @function_type_0 #bytecode.function_type<arguments = [], results = []>
// CHECK:  }
// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.func @fn @function_type_0 {
// CHECK:      [[REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG]], 0
// CHECK:      bytecode.assert [[REG]], @assert_msg_0
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

// Test ext.func -> func conversion with type decomposition
module {

bytecode.func_section @func_section {
  bytecode.ext.func @add (i64, i64) -> (i64) {
    %0 = bytecode.virtual_general_register
    bytecode.set_imm %0, 42
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @i64 i64
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@i64, @i64], results = [@i64]>
// CHECK:  }
// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.func @add @function_type_{{[0-9]+}} {
// CHECK:      [[REG:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG]], 42
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}

// -----

// Test type deduplication: same type used multiple times results in a single type section entry
module {

bytecode.func_section @func_section {
  bytecode.ext.func @identity (i64) -> (i64) {
    bytecode.ret
  }
}

// The i64 type should only appear once despite being used as both argument and result
// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @i64 i64
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@i64], results = [@i64]>
// CHECK:  }

}

// -----

// Test memref with float element type decomposition into buffer type
module {

bytecode.func_section @func_section {
  bytecode.ext.func @float_buffer_fn (memref<2x3x4xf32>) -> () {
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @f32 f32
// CHECK:    bytecode.type @buffer_type_{{[0-9]+}} #bytecode.buffer_type<element_type = @f32, rank = 3, shape = [2, 3, 4]
// CHECK-SAME: strides = [12, 4, 1]>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@buffer_type_{{[0-9]+}}], results = []>
// CHECK:  }

}

// -----

// Test E4M3 float format mapping
module {

bytecode.func_section @func_section {
  bytecode.ext.func @e4m3_fn (f8E4M3FN) -> () {
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @f8_e4m3 f8E4M3FN
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@f8_e4m3], results = []>
// CHECK:  }

}

// -----

// Test E5M2 float format mapping
module {

bytecode.func_section @func_section {
  bytecode.ext.func @e5m2_fn (f8E5M2) -> () {
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @f8_e5m2 f8E5M2
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@f8_e5m2], results = []>
// CHECK:  }

}

// -----

// Test memref type decomposition into buffer type
module {

bytecode.func_section @func_section {
  bytecode.ext.func @buffer_fn (memref<1x16x32x32xi64>) -> () {
    bytecode.ret
  }
}

// CHECK:  bytecode.type_section @type_section {
// CHECK:    bytecode.type @i64 i64
// CHECK:    bytecode.type @buffer_type_{{[0-9]+}} #bytecode.buffer_type<element_type = @i64, rank = 4, shape = [1, 16, 32, 32]
// CHECK-SAME: strides = [16384, 1024, 32, 1]>
// CHECK:    bytecode.type @function_type_{{[0-9]+}} #bytecode.function_type<arguments = [@buffer_type_{{[0-9]+}}], results = []>
// CHECK:  }

}
