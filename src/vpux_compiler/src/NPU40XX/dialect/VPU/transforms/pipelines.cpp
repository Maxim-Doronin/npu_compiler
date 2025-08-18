//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPU::arch40xx::buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                                                   Logger log) {
    pm.addPass(VPU::arch37xx::createDecomposeMVNPass(log));

    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(options.enablePrefetching, options.opTilingCacheThreshold,
                                                             options.mcOptimizationScope, log));
    pm.addPass(VPU::createConvertNCEInterpolateToDWPass(log));
    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation,
                                                  options.dumpStrategyToLog, false, log));
    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::arch37xx::createApplyTilingMVN1SumPass(options.enablePrefetching, log));
    pm.addPass(VPU::createTileLSTMSequencePass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(true, log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));

    VPU::buildTilingPipeline(pm, VPU::TilingOptions(options), log);

    if (options.enableScfComputeOpsOutlining) {
        pm.addPass(VPU::createScfComputeOpsOutliningPass(log));
        pm.addPass(VPU::createConvertDynamicToStaticKernelsPass(log));
    }

    pm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));
    pm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));

    pm.addPass(VPU::createComputeInterpolateCoordinatesPass(options.enableExplicitDistributionInfoAttr, log));
    pm.addPass(VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(log));

    pm.addPass(VPU::createMakeDistributedCopiesPass(log));
    pm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));
}

//
// DefaultHWPipeline
//

void vpux::VPU::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                 const VPU::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    /*
        Memory reservation for CMX has to happen as early in VPU as possible. It is required because memory reservation
        decreases usable CMX size which can result in different tiling decisions. If different passes see different
        effective CMX size different failures which can be hard to diagnose can happen. Examples of such failures
       include:
        - Fail during compilation if additional memory was reserved after tiling but before scheduling since tiles
        selected by tiling pipeline won't fit CMX anymore
        - Memory corruption if additional memory is reserved after scheduler since additional memory will overlap
        addresses allocated by the scheduler Currently there is no validation if memory is not reserved before the first
        call to getTotalCMXSize.
    */
    if (options.enableCompressActivationSpill) {
        pm.addPass(VPU::createCompressDmaReserveMemPass(log));
    }

    // Unconditional on NPU40xx due to DMA HWP scratch range requirement
    pm.addPass(VPU::createDMATaskProfilingReserveMemPass(
            options.enableProfiling ? options.enableDMAProfiling.getValue() : "false", log));

    /*
        Call this pass after all other memory reservation has already been done. This pass checks if there is 1KiB
        of reserved memory at the end of CMX and extends it if some is missing. So to not waste CMX memory make sure
        as much as possible is allocated in that 1KiB region. Exception to this rule is memory reserved for SW kernel IO
        for such memory make sure to reserve it after this pass to allow data prefetching.
    */
    pm.addPass(VPU::createSWKernelDataPrefetchReserveMemPass(log));

    // Make sure to run this after SWKernelDataPrefetchReserveMem which ensures we have enough
    // memory at the end of CMX to allow SW kernel data prefetch.
    // LNL Shave Kernel prefetch with profiling fails compiling. Track Number: E#169656
    if (options.enableSWKernelInstructionPrefetch && !(options.enableProfiling && options.enableSWProfiling)) {
        pm.addPass(VPU::createSWKernelInstructionPrefetchReserveMemForDummyKernelsPass(log));
    }

    // TODO: E#140041 enable profiling with outlining
    if (options.enableConcatRepeatingBlockOutlining && !options.enableProfiling) {
        pm.addPass(VPU::createConcatRepeatingBlocksOutliningPass(options.concatRepeatingBlockOutliningSeqLength, log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    pm.addPass(VPU::createTileGatherPass(log));
    pm.addPass(VPU::createConvertOpToDMAForPerformantExecutionPass(log));
    pm.addPass(VPU::arch40xx::createMoveConvertAroundViewLikeOpsPass(log));
    pm.addPass(VPU::arch37xx::createAdjustForOptimizedLayersPass(log));
    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::arch37xx::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::createAddSwOpAuxiliaryBufferPass(log));
    pm.addPass(VPU::createAdjustLSTMCellInputsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPU::createSplitSEOpsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
        pm.addPass(VPU::createLowerOpsToSENCEPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    }

    pm.addPass(VPU::createFuseClampPass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(options.enableOutputEnsurance, log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));
    if (options.enableWeightsSparsity) {
        VPU::buildWeightsSparsityPipeline(pm, VPU::WeightsSparsityOptions(options), log);
    }
    pm.addPass(VPU::createAddExplicitPaddingBeforeNCEPermutePass(log));
    if (VPU::isActSparsityEnabled(options.enableActivationSparsity)) {
        VPU::buildActivationSparsityPipeline(pm, VPU::ActivationSparsityOptions(options), log);
        pm.addPass(VPU::createLowerSparsityOpsPass(/*fakeSparsify=*/false, log));
    }

    if (options.enableInPlaceEltwise) {
        pm.addPass(VPU::createDetectInPlaceEltwisePass(log));
    }

    if (options.enableM2I) {
        pm.addPass(VPU::arch40xx::createFuseM2IOpsPass(log));
        pm.addPass(VPU::arch40xx::createConvertM2IOpsPass(log));
    }

    pm.addPass(VPU::createCostModelAnalysisConstructPass(log));
    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    } else {
        VPU::arch40xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    }

    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(VPU::createOptimizeSharedInputCopyForConcatPass(log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                             options.disablePassOnEntryFunctionForHostCompile, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPU::createMoveReflectPadToCMXPass(log));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::arch40xx::createCorrectNCEWorkloadsPass(log));
    pm.addPass(VPU::createResolveEltwiseWithZTiledWorkloadsPass(log));
    pm.addPass(VPU::arch40xx::createComputeNCEInputWorkloadsPass(log));
    pm.addPass(VPU::createShiftOutputWorkloadsForHaloPass(log));
    if (options.enableEntireMainContentOutlining) {
        pm.addPass(VPU::createOutlineEntireMainContentPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::VPU::arch40xx::registerVPUPipelines() {
    mlir::PassPipelineRegistration<VPU::arch40xx::DefaultHWOptions>(
            "default-hw-mode-vpu", "VPU dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPU::arch40xx::DefaultHWOptions& options) {
                VPU::arch40xx::buildDefaultHWPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<vpux::arch40xx::MCAndTilingOptionsDevice>(
            "incremental-pipeline", "Apply Incremental Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch40xx::MCAndTilingOptionsDevice& options) {
                VPU::arch40xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });

    mlir::PassPipelineRegistration<vpux::arch40xx::MCAndTilingOptionsDevice>(
            "sm-pipeline", "Apply SM Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch40xx::MCAndTilingOptionsDevice& options) {
                VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });
}
