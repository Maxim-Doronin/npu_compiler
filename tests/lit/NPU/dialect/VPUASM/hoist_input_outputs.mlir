//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" -allow-unregistered-dialect --hoist-input-outputs %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @hoistIO {

net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1xui8>
} outputsInfo : {
    DataInfo "prob" : tensor<2xf16>
    DataInfo "age_conv3" : tensor<3xf32>
} profilingOutputsInfo : {
    DataInfo "profilingOutput" {
    } : tensor<4xui64>
}

func.func @main(%arg0: memref<1xui8>, %arg1: memref<2xf16>, %arg2: memref<3xf32>, %arg3: memref<4xui64>) -> (memref<2xf16>, memref<3xf32>, memref<4xui64>) {

  "foo"(%arg0, %arg1) : (memref<1xui8>,  memref<2xf16>) -> ()
  "bar"(%arg2, %arg3) : (memref<3xf32>, memref<4xui64>) -> ()
  "foobar"(%arg0, %arg1, %arg2, %arg3) : (memref<1xui8>,  memref<2xf16>, memref<3xf32>, memref<4xui64>) -> ()

  VPUMI40XX.OpRanges
}
}

//CHECK:      VPUASM.InputBindings inputDeclarations
//CHECK-NEXT:   VPUASM.DeclareBuffer @data_buffDecl_0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1xui8> :  swizzling(0)>
//CHECK:      VPUASM.OutputBindings outputDeclarations
//CHECK-NEXT:   VPUASM.DeclareBuffer @prob_buffDecl_0 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<2xf16> :  swizzling(0)>
//CHECK-NEXT:   VPUASM.DeclareBuffer @age_conv3_buffDecl_1 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<3xf32> :  swizzling(0)>
//CHECK:      VPUASM.ProfilingBindings profilingDeclarations
//CHECK-NEXT:   VPUASM.DeclareBuffer @profilingOutput_buffDecl_0 !VPUASM.Buffer< "ProfilingOutput"[0] <0> : memref<4xui64> :  swizzling(0)>

// CHECK: func.func @main() {
// CHECK: [[ARG0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1xui8>
// CHECK: [[ARG1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<2xf16>
// CHECK: [[ARG2:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<3xf32>
// CHECK: [[ARG3:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<4xui64>
// CHECK: "foo"([[ARG0]], [[ARG1]])
// CHECK: "bar"([[ARG2]], [[ARG3]])
// CHECK: "foobar"([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]])

// CHECK: VPUMI40XX.OpRanges

// -----

module @hoistDuplicatedIO {
    net.NetworkInfo entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1xui8>
            DataInfo "data" : tensor<1xui8>
        } outputsInfo : {
            DataInfo "data" : tensor<1xui8>
            DataInfo "prob" : tensor<2xf16>
            DataInfo "prob" : tensor<2xf16>
            DataInfo "age_conv3" : tensor<3xf32>
        } profilingOutputsInfo : {
            DataInfo "data" : tensor<1xui8>
            DataInfo "profilingOutput" : tensor<4xui64> {
        }
    }

    func.func @main(%arg0: memref<1xui8>, %arg00: memref<1xui8>, %arg10: memref<1xui8>, %arg1: memref<2xf16>, %arg11: memref<2xf16>, %arg2: memref<3xf32>, %arg20: memref<1xui8>, %arg3: memref<4xui64>) -> (memref<1xui8>, memref<2xf16>, memref<2xf16>, memref<3xf32>, memref<1xui8>, memref<4xui64>) {
        "foo"(%arg0, %arg1) : (memref<1xui8>,  memref<2xf16>) -> ()
        "foo"(%arg00, %arg10) : (memref<1xui8>,  memref<1xui8>) -> ()
        "bar"(%arg2, %arg3) : (memref<3xf32>, memref<4xui64>) -> ()
        "bar"(%arg11, %arg3) : (memref<2xf16>, memref<4xui64>) -> ()
        "bar"(%arg20, %arg3) : (memref<1xui8>, memref<4xui64>) -> ()
        "foobar"(%arg0, %arg1, %arg2, %arg3) : (memref<1xui8>,  memref<2xf16>, memref<3xf32>, memref<4xui64>) -> ()

        VPUMI40XX.OpRanges
    }
}

// CHECK:      VPUASM.InputBindings inputDeclarations
// CHECK-NEXT:   VPUASM.DeclareBuffer @data_buffDecl_0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1xui8> :  swizzling(0)>
// CHECK-NEXT:   VPUASM.DeclareBuffer @data_buffDecl_1 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1xui8> :  swizzling(0)>

// CHECK:      VPUASM.OutputBindings outputDeclarations
// CHECK-NEXT:   VPUASM.DeclareBuffer @data_buffDecl_0 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1xui8> :  swizzling(0)>
// CHECK-NEXT:   VPUASM.DeclareBuffer @prob_buffDecl_1 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<2xf16> :  swizzling(0)>
// CHECK-NEXT:   VPUASM.DeclareBuffer @prob_buffDecl_2 !VPUASM.Buffer< "NetworkOutput"[2] <0> : memref<2xf16> :  swizzling(0)>
// CHECK-NEXT:   VPUASM.DeclareBuffer @age_conv3_buffDecl_3 !VPUASM.Buffer< "NetworkOutput"[3] <0> : memref<3xf32> :  swizzling(0)>

// CHECK:      VPUASM.ProfilingBindings profilingDeclarations
// CHECK-NEXT:   VPUASM.DeclareBuffer @data_buffDecl_0 !VPUASM.Buffer< "ProfilingOutput"[0] <0> : memref<1xui8> :  swizzling(0)>
// CHECK-NEXT:   VPUASM.DeclareBuffer @profilingOutput_buffDecl_1 !VPUASM.Buffer< "ProfilingOutput"[1] <0> : memref<4xui64> :  swizzling(0)>

// CHECK: func.func @main()

// CHECK: [[ARG0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1xui8>
// CHECK: [[ARG00:%.+]] = VPURT.DeclareBuffer <NetworkInput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1xui8>
// CHECK: [[ARG10:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1xui8>
// CHECK: [[ARG1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<2xf16>
// CHECK: [[ARG11:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [2] <0> {swizzlingKey = 0 : i64} -> memref<2xf16>
// CHECK: [[ARG2:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [3] <0> {swizzlingKey = 0 : i64} -> memref<3xf32>
// CHECK: [[ARG20:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1xui8>
// CHECK: [[ARG3:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<4xui64>

// CHECK: "foo"([[ARG0]], [[ARG1]])
// CHECK: "foo"([[ARG00]], [[ARG10]])
// CHECK: "bar"([[ARG2]], [[ARG3]])
// CHECK: "bar"([[ARG11]], [[ARG3]])
// CHECK: "bar"([[ARG20]], [[ARG3]])
// CHECK: "foobar"([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]])
