//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --outline-linalg-sw-layers | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3)>
module @NoBitcastUI  {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1xui32>
    DataInfo "input1" : tensor<1x1x1x16xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xui32>
  }

// CHECK: module @NoBitcastUI
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x1xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>

  func.func @main(%arg0: tensor<1x1x16x1xui32>, %arg1: tensor<1x1x16xui32>) -> tensor<1x1x16x16xui32> {
    %0 = tensor.bitcast %arg0 : tensor<1x1x16x1xui32> to tensor<1x1x16x1xi32>
    %1 = tensor.bitcast %arg1 : tensor<1x1x16xui32> to tensor<1x1x16xi32>
    %2 = tensor.empty() : tensor<1x1x16x16xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %1 : tensor<1x1x16x1xi32>, tensor<1x1x16xi32>) outs(%2 : tensor<1x1x16x16xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %12 = arith.maxui %in, %in_0 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x16x16xi32>
    %4 = tensor.bitcast %arg1 : tensor<1x1x16xui32> to tensor<1x1x16xi32>
    %5 = tensor.empty() : tensor<1x1x16x16xi32>
    %6 = linalg.generic {indexing_maps = [#NCHW, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<1x1x16x16xi32>, tensor<1x1x16xi32>) outs(%5 : tensor<1x1x16x16xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %12 = arith.divui %in, %in_0 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x16x16xi32>
    %7 = tensor.bitcast %arg1 : tensor<1x1x16xui32> to tensor<1x1x16xi32>
    %8 = tensor.empty() : tensor<1x1x16x16xi32>
    %9 = linalg.generic {indexing_maps = [#NCHW, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %7 : tensor<1x1x16x16xi32>, tensor<1x1x16xi32>) outs(%8 : tensor<1x1x16x16xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %12 = arith.minui %in, %in_0 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x16x16xi32>
    %10 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %6 : tensor<1x1x16x16xi32>, tensor<1x1x16x16xi32>) outs(%9 : tensor<1x1x16x16xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %12 = arith.maxui %in, %in_0 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x16x16xi32>
    %11 = tensor.bitcast %10 : tensor<1x1x16x16xi32> to tensor<1x1x16x16xui32>
    return %11 : tensor<1x1x16x16xui32>
// CHECK:         func.func @main([[ARG0:%.+]]: tensor<1x1x16x1xui32>, [[ARG1:%.+]]: tensor<1x1x16xui32>) -> tensor<1x1x16x16xui32>
// CHECK-NOT:     tensor.bitcast
// CHECK-NOT:     tensor.empty
// CHECK-NEXT:    [[R0:%.+]] = VPU.GenericSwLayer([[ARG0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x1xui32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xi32>
// CHECK-NEXT:    [[R1:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xi32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xi32>
// CHECK-NEXT:    [[R2:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xi32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xi32>
// CHECK-NEXT:    [[R3:%.+]] = VPU.GenericSwLayer([[R2]], [[R1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xi32>, tensor<1x1x16x16xi32> -> tensor<1x1x16x16xui32>
// CHECK-NEXT:    return [[R3]] : tensor<1x1x16x16xui32>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3)>
module @NoBitcastSI  {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1xsi32>
    DataInfo "input1" : tensor<1x1x1x16xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xsi32>
  }
// CHECK: module @NoBitcastSI
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x1xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>

  func.func @main(%arg0: tensor<1x1x16x1xsi32>, %arg1: tensor<1x1x16xsi32>) -> tensor<1x1x16x16xsi32> {
    %0 = tensor.bitcast %arg0 : tensor<1x1x16x1xsi32> to tensor<1x1x16x1xi32>
    %1 = tensor.bitcast %arg1 : tensor<1x1x16xsi32> to tensor<1x1x16xi32>
    %2 = tensor.empty() : tensor<1x1x16x16xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %1 : tensor<1x1x16x1xi32>, tensor<1x1x16xi32>) outs(%2 : tensor<1x1x16x16xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %12 = arith.maxsi %in, %in_0 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x16x16xi32>
    %4 = tensor.bitcast %arg1 : tensor<1x1x16xsi32> to tensor<1x1x16xi32>
    %5 = tensor.empty() : tensor<1x1x16x16xi32>
    %6 = linalg.generic {indexing_maps = [#NCHW, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<1x1x16x16xi32>, tensor<1x1x16xi32>) outs(%5 : tensor<1x1x16x16xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %12 = arith.divsi %in, %in_0 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x16x16xi32>
    %7 = tensor.bitcast %arg1 : tensor<1x1x16xsi32> to tensor<1x1x16xi32>
    %8 = tensor.empty() : tensor<1x1x16x16xi32>
    %9 = linalg.generic {indexing_maps = [#NCHW, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %7 : tensor<1x1x16x16xi32>, tensor<1x1x16xi32>) outs(%8 : tensor<1x1x16x16xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %12 = arith.minsi %in, %in_0 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x16x16xi32>
    %10 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %6 : tensor<1x1x16x16xi32>, tensor<1x1x16x16xi32>) outs(%9 : tensor<1x1x16x16xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %12 = arith.maxsi %in, %in_0 : i32
      linalg.yield %12 : i32
    } -> tensor<1x1x16x16xi32>
    %11 = tensor.bitcast %10 : tensor<1x1x16x16xi32> to tensor<1x1x16x16xsi32>
    return %11 : tensor<1x1x16x16xsi32>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3)>
module @NoBitcastF {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1xf16>
    DataInfo "input1" : tensor<1x1x1x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xf16>
  }

// CHECK: module @NoBitcastF
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xf16>, {{.*}}: tensor<1x1x16x16xf16>, {{.*}}: tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xf16>, {{.*}}: tensor<1x1x16xf16>, {{.*}}: tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xf16>, {{.*}}: tensor<1x1x16xf16>, {{.*}}: tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x1xf16>, {{.*}}: tensor<1x1x16xf16>, {{.*}}: tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>

  func.func @main(%arg0: tensor<1x1x16x1xf16>, %arg1: tensor<1x1x16xf16>) -> tensor<1x1x16x16xf16> {
    %0 = tensor.empty() : tensor<1x1x16x16xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x1x16x1xf16>, tensor<1x1x16xf16>) outs(%0 : tensor<1x1x16x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %7 = arith.maximumf %in, %in_0 fastmath<nnan,nsz> : f16
      linalg.yield %7 : f16
    } -> tensor<1x1x16x16xf16>
    %2 = tensor.empty() : tensor<1x1x16x16xf16>
    %3 = linalg.generic {indexing_maps = [#NCHW, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg1 : tensor<1x1x16x16xf16>, tensor<1x1x16xf16>) outs(%2 : tensor<1x1x16x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %7 = arith.divf %in, %in_0 fastmath<arcp> : f16
      linalg.yield %7 : f16
    } -> tensor<1x1x16x16xf16>
    %4 = tensor.empty() : tensor<1x1x16x16xf16>
    %5 = linalg.generic {indexing_maps = [#NCHW, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg1 : tensor<1x1x16x16xf16>, tensor<1x1x16xf16>) outs(%4 : tensor<1x1x16x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %7 = arith.minimumf %in, %in_0 fastmath<nnan,nsz> : f16
      linalg.yield %7 : f16
    } -> tensor<1x1x16x16xf16>
    %6 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %3 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) outs(%5 : tensor<1x1x16x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %7 = arith.maximumf %in, %in_0 fastmath<nnan,nsz> : f16
      linalg.yield %7 : f16
    } -> tensor<1x1x16x16xf16>
    return %6 : tensor<1x1x16x16xf16>

// CHECK:         func.func @main([[ARG0:%.+]]: tensor<1x1x16x1xf16>, [[ARG1:%.+]]: tensor<1x1x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK-NOT:     tensor.bitcast
// CHECK-NOT:     tensor.empty
// CHECK-NEXT:    [[R0:%.+]] = VPU.GenericSwLayer([[ARG0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x1xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[R1:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[R2:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[R3:%.+]] = VPU.GenericSwLayer([[R2]], [[R1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16> -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    return [[R3]] : tensor<1x1x16x16xf16>
  }
}
