//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// UnrollDistributedOps
//

def UnrollDistributedOps : PassBase<"unroll-distributed-ops", "vpux::FunctionPass"> {
    let summary = "Extracts child operations from distributed operation";

    let description = [{
        Legalizes chain of distributed operations according to the tile schema
    }];

    let constructor = "vpux::VPUIP::arch37xx::createUnrollDistributedOpsPass()";
}

//
// AddSwKernelCacheHandlingOps
//

def AddSwKernelCacheHandlingOps : PassBase<"add-sw-kernel-cache-handling-ops", "vpux::FunctionPass"> {
    let summary = "Add cache-handling ops for SW Kernels";

    let description = [{
        This pass adds CACHE_FLUSH or CACHE_FLUSH_INVALIDATE when needed.

        CACHE_FLUSH will be added when at least one output buffer is located in DDR and at least one of its
        consumers is not an ActSHAVE such that the ActSHAVE cache is flushed after the ActSHAVE Kernel writes
        to DDR through the cache.

        CACHE_FLUSH_INVALIDATE will be added when at least one input buffer is located in DDR and it is not
        produced by an ActSHAVE or is a constant, such that the ActSHAVE cache is invalidated before the
        ActSHAVE Kernel reads from DDR through the cache.

        If there are two consecutive SW Kernels and a CACHE_FLUSH was added after the first one and a CACHE_INVALIDATE
        is needed before the second one, the resulting sequence of ops will be:
        [SW Kernel 1] - [CACHE_FLUSH] - [CACHE_FLUSH_INVALIDATE] - [SW Kernel 2].

        CACHE_INVALIDATE no longer used since it's possible that the SHAVE stack resides in DDR, in which case
        a CACHE_INVALIDATE on the Shave L2 cache causes the corruption of the SHAVE stack. The SHAVE jumps to a
        null address on completion of the L2 invalidate.
    }];

    let constructor = "vpux::VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass()";
}

//
// UnrollDepthToSpaceDMA
//

def UnrollDepthToSpaceDMA : PassBase<"unroll-depth-to-space-dma", "vpux::FunctionPass"> {
    let summary = "Split DepthToSpaceDMA task with several NN DMA tasks";

    let description = [{
        This pass spilt DepthToSpaceDMA tasks with several NN DMA tasks, which are functionally equivalent.
        Each sub DepthToSpaceDMA will be converted to a NNDMA.
        1. if input/output layout is NHWC with model block_first, number of sub DepthToSpaceDMA is same as block_size.
        2. if input/output layout is NHWC with model depth_first, number of sub DepthToSpaceDMA is OH * OW / block_size.
        block_size is the size of the spatial block. It is an attribution of DepthToSpace.
    }];

    let constructor = "vpux::VPUIP::arch37xx::createUnrollDepthToSpaceDMAPass()";
}

//
// UnrollSpaceToDepthDMA
//

def UnrollSpaceToDepthDMA : PassBase<"unroll-space-to-depth-dma", "vpux::FunctionPass"> {
    let summary = "Split SpaceToDepthDMA task with several NN DMA tasks";

    let description = [{
        This pass splits SpaceToDepthDMA tasks into several NN DMA tasks, which are functionally equivalent.
        Each sub SpaceToDepthDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::arch37xx::createUnrollSpaceToDepthDMAPass()";
}

//
// UnrollPermuteDMA
//

def UnrollPermuteDMA : PassBase<"unroll-permute-dma", "vpux::FunctionPass"> {
    let summary = "Unroll PermuteDMA tasks";

    let description = [{
        This pass unrolls PermuteDMA tasks to one or several PermuteDMA tasks.
        The number of new PermuteDMA tasks depends on the number of planes (num_planes <= 256).
        1. NCHW -> NHWC: The number of planes is C.
        2. NHWC -> NCHW: The number of planes is H * W, and W must be <= 256.
    }];

    let constructor = "vpux::VPUIP::arch37xx::createUnrollPermuteDMAPass()";
}

#endif
