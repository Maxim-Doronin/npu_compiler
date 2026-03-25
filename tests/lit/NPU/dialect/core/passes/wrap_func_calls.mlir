//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wrap-func-call --verify-diagnostics %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


// CHECK-LABEL: @NoWrapFuncModule
module @NoWrapFuncModule {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %call1 = call @foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %call1 : tensor<2x2xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[CALL1:%.+]] = call @foo([[ARG]])
    // CHECK:   return [[CALL1]]
}

// -----

// CHECK-LABEL: @SimpleWrapping
module @SimpleWrapping {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
        DataInfo "input_1" :tensor<3x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
        DataInfo "output_1" : tensor<3x2xf32>
    }

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo"]} {
        return %arg : tensor<2x2xf32>
    }

    func.func private @bar(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_bar"]} {
        return %arg : tensor<3x2xf32>
    }

    func.func private @real_foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    func.func private @real_bar(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> {
        return %arg : tensor<3x2xf32>
    }


    func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<3x2xf32>) -> (tensor<2x2xf32>, tensor<3x2xf32>) {
        %call1 = call @foo(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %call2 = call @bar(%arg1) : (tensor<3x2xf32>) -> tensor<3x2xf32>
        return %call1, %call2 : tensor<2x2xf32>, tensor<3x2xf32>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<2x2xf32>, [[ARG1:%.+]]: tensor<3x2xf32>) -> (tensor<2x2xf32>, tensor<3x2xf32>)
    // CHECK:   [[CALL1:%.+]] = call @real_foo([[ARG0]])
    // CHECK:   [[CALL2:%.+]] = call @real_bar([[ARG1]])
    // CHECK:   return [[CALL1]], [[CALL2]]
}

// -----


// CHECK-LABEL: @SimpleExternalFuncWrapping
module @SimpleExternalFuncWrapping {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
        DataInfo "input_1" :tensor<3x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
        DataInfo "output_1" : tensor<3x2xf32>
    }

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo"]}

    func.func private @bar(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_bar"]}

    func.func private @real_foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    func.func private @real_bar(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> {
        return %arg : tensor<3x2xf32>
    }


    func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<3x2xf32>) -> (tensor<2x2xf32>, tensor<3x2xf32>) {
        %call1 = call @foo(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %call2 = call @bar(%arg1) : (tensor<3x2xf32>) -> tensor<3x2xf32>
        return %call1, %call2 : tensor<2x2xf32>, tensor<3x2xf32>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<2x2xf32>, [[ARG1:%.+]]: tensor<3x2xf32>) -> (tensor<2x2xf32>, tensor<3x2xf32>)
    // CHECK:   [[CALL1:%.+]] = call @real_foo([[ARG0]])
    // CHECK:   [[CALL2:%.+]] = call @real_bar([[ARG1]])
    // CHECK:   return [[CALL1]], [[CALL2]]
}


// -----

// CHECK-LABEL: @SimpleWrappingMultipleOccurrences
module @SimpleWrappingMultipleOccurrences {

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo"]} {
        return %arg : tensor<2x2xf32>
    }

    func.func private @bar(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %ret = call @foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %ret : tensor<2x2xf32>
    }

    // CHECK: func.func private @bar([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[RET:%.+]] = call @real_foo([[ARG]])
    // CHECK:   return [[RET]]

    func.func private @baz(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %ret = call @foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %ret : tensor<2x2xf32>
    }

    // CHECK: func.func private @baz([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[RET:%.+]] = call @real_foo([[ARG]])
    // CHECK:   return [[RET]]

    func.func private @real_foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

}


// -----

// CHECK-LABEL: @CrossWrapping
module @CrossWrapping {

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo"]} {
        %call = call @bar(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %call : tensor<2x2xf32>
    }

    // CHECK: func.func private @foo([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[RET:%.+]] = call @real_bar([[ARG]])
    // CHECK:   return [[RET]]


    func.func private @bar(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_bar"]} {
        %call = call @foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %call : tensor<2x2xf32>
    }

    // CHECK: func.func private @bar([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[RET:%.+]] = call @real_foo([[ARG]])
    // CHECK:   return [[RET]]


    func.func private @real_foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    func.func private @real_bar(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }
}


// -----

// CHECK-LABEL: @CrossWrappingWithDelete
module @CrossWrappingWithDelete {

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo", "deleteWrapped=1"]} {
        %call = call @bar(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %call : tensor<2x2xf32>
    }

    func.func private @bar(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_bar", "deleteWrapped=1"]} {
        %call = call @foo(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %call : tensor<2x2xf32>
    }

    func.func private @real_foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    func.func private @real_bar(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    // CHECK-NOT: @foo
    // CHECK-NOT: @bar
}


// -----

// CHECK-LABEL: @SimpleWrappingDeleteOneWrapped
module @SimpleWrappingDeleteOneWrapped {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input_0": tensor<2x2xf32>
        DataInfo "input_1" :tensor<3x2xf32>
    } outputsInfo : {
        DataInfo "output_0" : tensor<2x2xf32>
        DataInfo "output_1" : tensor<2x2xf32>
        DataInfo "output_2" : tensor<3x2xf32>
        DataInfo "output_3" : tensor<3x2xf32>
    }

    func.func private @foo_boolean_attr(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo", "deleteWrapped=false"]} {
        return %arg : tensor<2x2xf32>
    }

    func.func private @foo_int_attr(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo", "deleteWrapped=0"]} {
        return %arg : tensor<2x2xf32>
    }

    func.func private @to_delete_boolean_attr(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_bar", "deleteWrapped=true"]} {
        return %arg : tensor<3x2xf32>
    }

    func.func private @to_delete_int_attr(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_bar", "deleteWrapped=1"]} {
        return %arg : tensor<3x2xf32>
    }

    func.func private @real_foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    func.func private @real_bar(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> {
        return %arg : tensor<3x2xf32>
    }

    func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<3x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>) {
        %call10 = call @foo_boolean_attr(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %call11 = call @foo_int_attr(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %call20 = call @to_delete_boolean_attr(%arg1) : (tensor<3x2xf32>) -> tensor<3x2xf32>
        %call21 = call @to_delete_int_attr(%arg1) : (tensor<3x2xf32>) -> tensor<3x2xf32>
        return %call10, %call11, %call20, %call21 : tensor<2x2xf32>, tensor<2x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>
    }

    // CHECK: foo_boolean_attr
    // CHECK: foo_int_attr
    // CHECK-NOT: to_delete_boolean_attr
    // CHECK-NOT: to_delete_int_attr

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<2x2xf32>, [[ARG1:%.+]]: tensor<3x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>)
    // CHECK:   [[CALL10:%.+]] = call @real_foo([[ARG0]])
    // CHECK:   [[CALL11:%.+]] = call @real_foo([[ARG0]])
    // CHECK:   [[CALL20:%.+]] = call @real_bar([[ARG1]])
    // CHECK:   [[CALL21:%.+]] = call @real_bar([[ARG1]])
    // CHECK:   return [[CALL10]], [[CALL11]], [[CALL20]], [[CALL21]]
}

// -----

// expected-error@+1 {{"not_found_real_bar" must exist in module. Required by the wrapped function: "bar"}}
module @WrapperNotFound {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
        DataInfo "input_1" :tensor<3x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
        DataInfo "output_1" : tensor<3x2xf32>
    }

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo"]} {
        return %arg : tensor<2x2xf32>
    }

    func.func private @bar(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {wrapFunctionAttr = ["wrapper=not_found_real_bar"]} {
        return %arg : tensor<3x2xf32>
    }

    func.func private @real_foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        return %arg : tensor<2x2xf32>
    }

    func.func private @real_bar(%arg: tensor<3x2xf32>) -> tensor<3x2xf32> {
        return %arg : tensor<3x2xf32>
    }

    func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<3x2xf32>) -> (tensor<2x2xf32>, tensor<3x2xf32>) {
        %call1 = call @foo(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %call2 = call @bar(%arg1) : (tensor<3x2xf32>) -> tensor<3x2xf32>
        return %call1, %call2 : tensor<2x2xf32>, tensor<3x2xf32>
    }
}

// -----

// expected-error@+1 {{The function wrapper: "real_foo" must have identical}}
module @WrappedWrapperTypesDiscrepancy {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {wrapFunctionAttr = ["wrapper=real_foo"]} {
        return %arg : tensor<2x2xf32>
    }

    func.func private @real_foo(%arg: memref<2x2xf32>) -> memref<2x2xf32> {
        return %arg : memref<2x2xf32>
    }

    func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %res = call @foo(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %res : tensor<2x2xf32>
    }
}
