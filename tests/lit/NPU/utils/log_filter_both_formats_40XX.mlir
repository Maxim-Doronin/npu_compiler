//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This test verifies that IE_NPU_LOG_FILTER accepts both pass name formats:
// - CamelCase: "DumpStatisticsOfIEOps" (from pass->getName())
// - dash-separated: "dump-statistics-of-ie-ops" (from pass->getArgument())

// Test with dash-separated format (original behavior)
// RUN: env OV_NPU_LOG_LEVEL=LOG_INFO IE_NPU_LOG_FILTER=dump-statistics-of-ie-ops vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dump-statistics-of-ie-ops -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-DASH

// Test with CamelCase format (new behavior)
// RUN: env OV_NPU_LOG_LEVEL=LOG_INFO IE_NPU_LOG_FILTER=DumpStatisticsOfIEOps vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dump-statistics-of-ie-ops -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-CAMEL

// REQUIRES: arch-NPU40XX

module @TestLogFilterFormats {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2xf16>
    }

    func.func @main(%input: tensor<2x1xf16>) -> tensor<1x2xf16> {
        %result = IE.Reshape(%input) {shape_value = [1, 2]} : tensor<2x1xf16> -> tensor<1x2xf16>
        return %result : tensor<1x2xf16>
    }

    // Both formats should produce the same statistics output
    // CHECK-DASH:   IE dialect statistics:
    // CHECK-DASH:   IE - 1
    // CHECK-DASH:     Non-computational - 1 (100.00%)
    // CHECK-DASH:       IE.Reshape - 1 (100.00%)

    // CHECK-CAMEL: IE dialect statistics:
    // CHECK-CAMEL: IE - 1
    // CHECK-CAMEL:   Non-computational - 1 (100.00%)
    // CHECK-CAMEL:     IE.Reshape - 1 (100.00%)
}
