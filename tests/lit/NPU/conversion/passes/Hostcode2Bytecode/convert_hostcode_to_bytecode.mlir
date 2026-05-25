//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-hostcode-to-bytecode %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module {

func.func @fn1(%arg0 : i64) -> () attributes {config.pureHostCompileFunc} {
  %c32f = arith.constant 32.0 : f64
  %c32i = arith.constant 32 : i64
  %add = arith.addi %arg0, %c32i : i64
  return
}
func.func @fn2() -> () attributes {config.pureHostCompileFunc} {
  %false = arith.constant false
  cf.assert %false, "Assertion failed"
  return
}
func.func @fn3(%arg0 : i64, %arg1 : i64) -> () attributes {config.pureHostCompileFunc} {
  %mul = arith.muli %arg0, %arg1 : i64
  %min = arith.minsi %arg0, %arg1 : i64
  %max = arith.maxsi %arg0, %arg1 : i64
  return
}

// CHECK:  bytecode.func_section @func_section {
// CHECK:    bytecode.ext.func @fn1 (i64) -> () {
// CHECK:      [[PARAM_REG:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_C32F:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG_C32F]], 4629700416936869888
// CHECK:      [[REG_C32I:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG_C32I]], 32
// CHECK:      [[REG_ADD:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.add.i64 [[REG_ADD]], [[PARAM_REG]], [[REG_C32I]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn2 () -> () {
// CHECK:      [[REG_FALSE:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.set_imm [[REG_FALSE]], 0
// CHECK:      bytecode.ext.assert [[REG_FALSE]], "Assertion failed"
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:    bytecode.ext.func @fn3 (i64, i64) -> () {
// CHECK:      [[PARAM1:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:      [[PARAM0:%.+]] = bytecode.virtual_parameter_register 0
// CHECK:      [[REG_MUL:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.mul.i64 [[REG_MUL]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MIN:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.min.i64 [[REG_MIN]], [[PARAM0]], [[PARAM1]]
// CHECK:      [[REG_MAX:%.+]] = bytecode.virtual_general_register
// CHECK:      bytecode.max.i64 [[REG_MAX]], [[PARAM0]], [[PARAM1]]
// CHECK:      bytecode.ret
// CHECK:    }
// CHECK:  }

}
