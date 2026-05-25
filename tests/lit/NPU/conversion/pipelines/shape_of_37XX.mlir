//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --platform=%platform% --import-IE ./shape_of.xml | FileCheck %s
// REQUIRES: platform-NPU3720

// CHECK: module @shape_of {
// CHECK:   func.func @main([[ARG_0:%[^:]+]]: tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>)
// CHECK-SAME:      -> tensor<4xsi64> {
// CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[ARG_0]]) {dstElemType = si64} :
// CHECK-SAME:      tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
// CHECK:       return [[SHAPE_OF]] : tensor<4xsi64>
// CHECK:   }
// CHECK: }
