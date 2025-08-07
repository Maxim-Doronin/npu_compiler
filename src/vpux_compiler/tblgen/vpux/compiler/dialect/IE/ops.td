//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS
#define VPUX_COMPILER_DIALECT_IE_OPS

include "vpux/compiler/dialect/IE/ops/activation.td"
include "vpux/compiler/dialect/IE/ops/arithmetic.td"
include "vpux/compiler/dialect/IE/ops/bitwise.td"
include "vpux/compiler/dialect/IE/ops/comparison.td"
include "vpux/compiler/dialect/IE/ops/control_flow.td"
include "vpux/compiler/dialect/IE/ops/convolution.td"
include "vpux/compiler/dialect/IE/ops/data_movement.td"
include "vpux/compiler/dialect/IE/ops/data_type.td"
include "vpux/compiler/dialect/IE/ops/eltwise.td"
include "vpux/compiler/dialect/IE/ops/image.td"
include "vpux/compiler/dialect/IE/ops/logical.td"
include "vpux/compiler/dialect/IE/ops/normalization.td"
include "vpux/compiler/dialect/IE/ops/pooling.td"
include "vpux/compiler/dialect/IE/ops/recurrent.td"
include "vpux/compiler/dialect/IE/ops/reduce.td"
include "vpux/compiler/dialect/IE/ops/resources.td"
include "vpux/compiler/dialect/IE/ops/shape_manipulation.td"
include "vpux/compiler/dialect/IE/ops/specialized.td"

//
// ExternalKernelOp
//

def IE_ExternalKernelOp :
        IE_Op<
            "ExternalKernel",
            []
        > {
    let summary = "Represents a kernel whose details/implementation are defined externally";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        DictionaryAttr:$attrDict,
        StrAttr:$unique_id,
        StrAttr:$kernel_path
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let assemblyFormat = [{
        $unique_id
        `at_path` `(` $kernel_path `)`
        (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
        `attrs` `(` $attrDict `)`
        attr-dict
        `->` type(results)
    }];
}

#endif
