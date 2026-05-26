//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

bytecode.type_section @type_section {
    bytecode.type @i64_type i64
    bytecode.type @f32_type f32
    // CHECK:  bytecode.type @i64_type i64
    // CHECK:  bytecode.type @f32_type f32
}
