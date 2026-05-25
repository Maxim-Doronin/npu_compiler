//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// Test that a valid type reference passes verification
module {
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    bytecode.func @valid_fn @fn_type {
        bytecode.ret
    }
}
}

// -----

// Test that a dangling type reference is caught by the verifier
module {
bytecode.type_section @type_section {
    bytecode.type @fn_type #bytecode.function_type<arguments = [], results = []>
}
bytecode.func_section @func_section {
    // expected-error @+1 {{could not be resolved in the type section}}
    bytecode.func @bad_fn @nonexistent {
        bytecode.ret
    }
}
}

// -----

// Test that a function type reference pointing to a non-function type is caught
module {
bytecode.type_section @type_section {
    bytecode.type @i64_type #bytecode.integer_type<width = 64>
}
bytecode.func_section @func_section {
    // expected-error @+1 {{resolves to a non-function type in the type section}}
    bytecode.func @bad_fn @i64_type {
        bytecode.ret
    }
}
}
