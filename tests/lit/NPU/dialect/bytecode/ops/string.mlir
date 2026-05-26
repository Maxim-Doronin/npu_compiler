//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

bytecode.string_section @string_section {
    bytecode.string @my_string "Example of a string"
    bytecode.string @another_string "Another example"
    // CHECK:  bytecode.string @my_string "Example of a string"
    // CHECK:  bytecode.string @another_string "Another example"
}
