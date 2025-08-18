//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_40XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// ReduceNumTilesForSmallModelsPass
//

def ReduceNumTilesForSmallModelsPass: PassBase<"reduce-num-tiles-for-small-models", "vpux::ModulePass"> {
    let summary = "Reduce number of clusters for small models";

    let description = [{
        There are some cases when running a model on all clusters can be slower than on one cluster.

        Based on observations, it happens on transformer models when the following conditions are met:
            1) The model's inference latency is small, e.g. few hundreds of microseconds.
            2) The shapes for most of the layers are small, e.g. ~10,000 elements.
            3) From these small layers there are a lot of Add & Multiply that are not fused into Convolution,
                so they end up being a separate DPU operation.

        Running such a model on all clusters introduces overhead, that is bigger than the improvement mostly because:
            1) Shave initialization takes longer, as we need to initialize more shaves.
            2) There are a lot of tasks (both compute and DMA), which creates pressure on scheduler.
            3) A lot of small DMA transactions create pressure on DMA unit.

        The pass makes a choice based on two main heuristics comparing them with experimental numbers:
            1) Mean and standard deviation of tensor sizes.
            2) Ratio of MatMul/Multiply/SoftMax operations (that are frequently met in transformer models).
    }];

    let constructor = "vpux::IE::arch40xx::createReduceNumTilesForSmallModelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
