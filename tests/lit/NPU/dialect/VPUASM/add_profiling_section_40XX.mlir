//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --add-profiling-section %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @AddProfilingSection {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x2x3x4xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2x3x4xf16>
    } profilingOutputsInfo : {
    }
    func.func @main() {
        ELF.Main {
            VPUASM.ProfilingMetadata @ProfilingMetadata {metadata = dense<1> : vector<184xui8>}
        }
        return
    }
}

// CHECK: ELF.CreateProfilingSection @".profiling" aligned(1) secFlags(SHF_ALLOC) {
// CHECK-NEXT: VPUASM.ProfilingMetadata @ProfilingMetadata {metadata = {{.+}} : vector<184xui8>}
