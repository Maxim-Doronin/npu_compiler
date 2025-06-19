//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_ARCH_40XX_PASSES
#define VPUX_COMPILER_DIALECT_VPURT_ARCH_40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// InsertSyncTasks
//

def InsertSyncTasks : PassBase<"insert-sync-tasks", "vpux::FunctionPass"> {
    let summary = "Insert sync tasks";

    let description = [{
        Insert sync task that mark start and end of each FuncOp.
        Example:

               | op -> .. -> op |
        funcOp{| op -> .. -> op |}
               | op -> .. -> op |

                        =>

               |          /-> op -> .. -> op -\           |
        funcOp{| SyncTask --> op -> .. -> op --> SyncTask |}
               |          \-> op -> .. -> op -/           |

    }];

    let constructor = "vpux::VPURT::arch40xx::createInsertSyncTasksPass()";
}

//
// OptimizeSyncTasks
//

def OptimizeSyncTasks : PassBase<"optimize-sync-tasks", "vpux::FunctionPass"> {
    let summary = "Remove unnecessary sync tasks";

    let description = [{
        Remove unnecessary sync task that were injected previously by InsertSyncTasks pass.
        Example:

                 /-> op -> .. -> op -\                       /-> op -> .. -> op -\
        SyncTask --> op -> .. -> op --> SyncTask -> SyncTask --> op -> .. -> op --> SyncTask
                 \-> op -> .. -> op -/                       \-> op -> .. -> op -/

                        =>

        /-> op -> .. -> op -\           /-> op -> .. -> op -\
        --> op -> .. -> op --> SyncTask --> op -> .. -> op -->
        \-> op -> .. -> op -/           \-> op -> .. -> op -/
    }];

    let constructor = "vpux::VPURT::arch40xx::createOptimizeSyncTasksPass()";
}

//
// FindWlmEnqueueBarrier
//

def FindWlmEnqueueBarrier : PassBase<"find-wlm-enqueue-barrier", "vpux::FunctionPass"> {
    let summary = "For WLM for each task find enqueue barrier";

    let description = [{
        For WLM each task needs to be enqueued at some point. This pass analyzes schedule
        and for each task finds a barrier at which it can be enqueued. Later this information
        will be lowered to dedicated enqueue ops (WorkItem task)
    }];

    let constructor = "vpux::VPURT::arch40xx::createFindWlmEnqueueBarrierPass()";

    let options = [
        Option<
            "enqDmaAtBootstrapOpt", "enq-dma-at-bootstrap",
            "bool", "",
            "Enforce enqueuing DMA at bootstrap"
        >
    ];
}

//
// OrderBarriersForWlm
//

def OrderBarriersForWlm : PassBase<"order-barriers-for-wlm", "vpux::FunctionPass"> {
    let summary = "Order barriers for WLM";

    let description = [{
        For WLM barriers are epxected to be ordered following consumption order
    }];

    let constructor = "vpux::VPURT::arch40xx::createOrderBarriersForWlmPass()";
}


//
// WlmSplitGraphToPages
//

def WlmSplitGraphToPages : PassBase<"wlm-split-graph-to-pages", "vpux::FunctionPass"> {
    let summary = "For WLM split model into subgraphs (pages)";

    let description = [{
        This pass is intended for WLM mode where model is split into subgraphs called pages,
        limited by number of barriers, to allow later seamless enabling of WLM related features.
    }];

    let constructor = "vpux::VPURT::arch40xx::createWlmSplitGraphToPagesPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

//
// WlmLegalizeSplitGraphToPages
//

def WlmLegalizeSplitGraphToPages : PassBase<"wlm-legalize-split-graph-to-pages", "vpux::FunctionPass"> {
    let summary = "For WLM legalize split model into subgraphs (pages)";

    let description = [{
        This pass is intended to legalize split into subgraphs for WLM
    }];

    let constructor = "vpux::VPURT::arch40xx::createWlmLegalizeSplitGraphToPagesPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}


//
// WlmLegalizePagesForBarrierDmas
//

def WlmLegalizePagesForBarrierDmas : PassBase<"wlm-legalize-pages-for-barrier-dmas", "vpux::FunctionPass"> {
    let summary = "For WLM legalize split into pages for reprogramming Barrier FIFOs using compiler DMAs";

    let description = [{
        This pass is intended to legalize split into pages for WLM in case compiler will be responsible
        for programming virtual instances of physical barriers by writing to the Barrier FIFO using DMAs
        During this pass, we not only adjust the schedule but also insert reference DMAs
        These reference DMAs serve as placeholders and will later be replaced with actual DMAs that will
        write barrier descriptors to the Barrier FIFO registers
    }];

    let constructor = "vpux::VPURT::arch40xx::createWlmLegalizePagesForBarrierDmasPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >,
        Option<
            "barrierFifoDepthOpt", "barrier-fifo-depth",
            "int", "",
            "Depth of barrier fifo"
        >
    ];
}

//
// WlmInsertDummyDmasInPages
//

def WlmInsertDummyDmasInPages : PassBase<"wlm-insert-dummy-dmas-in-pages", "vpux::FunctionPass"> {
    let summary = "For WLM pages split insert dummy DMAs if given page doesn't have this DMA type";

    let description = [{
        This pass checks if in each page there is at least 1 DMA of each type to enable possibility
        of enqueuing all DMAs at bootstrap as single link-list of tasks. If there is no DMAx in PageN
        that guarantees that barriers from PageN-1 have been consumed and there are DMAs of this type
        in some later pages, then we need to insert task of DMAx in PageN and this DMAx
        needs to depend on all boundary tasks on PageN-1/PageN border.
    }];

    let constructor = "vpux::VPURT::arch40xx::createWlmInsertDummyDmasInPagesPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

#endif
