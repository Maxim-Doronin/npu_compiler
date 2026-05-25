//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

bytecode.kernel_section @kernel_section {
    bytecode.kernel @first_kernel "\00\01\02\03"
    bytecode.kernel @second_kernel "\04\05\06\07\08"
    // CHECK:  bytecode.kernel @first_kernel "\00\01\02\03"
    // CHECK:  bytecode.kernel @second_kernel "\04\05\06\07\08"
}
