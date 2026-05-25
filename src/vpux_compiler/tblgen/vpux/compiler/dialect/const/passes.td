//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONST_PASSES
#define VPUX_COMPILER_DIALECT_CONST_PASSES

include "mlir/Pass/PassBase.td"

//
// ConstantFolding
//

def ConstantFolding : PassBase<"constant-folding", "vpux::FunctionPass"> {
    let summary = "Constant folding";

    let description = [{
        AI-generated summary

        Materializes Const.DeclareOp transformation pipelines (quantize, dequantize,
        reshape, transpose, pad, etc.) into final folded constant data.
        Constants smaller than 1 KB are folded inline; larger constants are folded
        in parallel threads. Constants exceeding a configurable size threshold
        (default 300 MB) are folded on a single thread to limit peak memory usage.
    }];

    let constructor = "vpux::Const::createConstantFoldingPass()";

    let dependentDialects = [
        "vpux::Const::ConstDialect"
    ];
}

def ApplySwizzling : PassBase<"apply-swizzling", "vpux::FunctionPass"> {
    let summary = "apply swizzling transform for swizzled constants";

    let description = [{
        This pass will apply swizzling content transform to constants
        with swizzlingSchemeAttr in its type. We can't attach Swizzling
        transformation earlier during swizzling pass since other passes
        might want to add more transformations to unswizzled data.
        Since inserting transformations before swizzling transform will
        worsen background folding performance this pass will add swizzling
        transformation once we are late enough in the compilation flow to
        be sure that no other pass wants to insert transformation before swizzling.
    }];

    let constructor = "vpux::Const::createApplySwizzlingPass()";

    let dependentDialects = [
        "vpux::Const::ConstDialect"
    ];
}

#endif
