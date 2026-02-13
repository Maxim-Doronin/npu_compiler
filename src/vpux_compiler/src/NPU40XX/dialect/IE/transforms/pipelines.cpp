//
// Copyright (C) 2023-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/locverif/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// Outlining
//

void vpux::IE::arch40xx::buildOutliningPipeline(mlir::OpPassManager& pm, const DefaultHWOptionsBase& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Blob compilation using 'debatcher' method leverages 'outlining' feature so that
    // it will be turned on unless it was already enabled
    bool isOutliningEnabled = options.functionOutlining.hasValue() || DebatcherOptions::isAvailable(options);
    if (!canOutlineFromProfilingPerspective(options)) {
        // TODO: E#140041 enable profiling with outlining
        log.warning("Outlining was disabled due to profiling being enabled");
        isOutliningEnabled = false;
    }
    if (isOutliningEnabled) {
        if (options.enableLoopOutliner) {
            pm.addPass(IE::createLoopOutlinerPass(log));
        }
        pm.addPass(mlir::createCanonicalizerPass(grc));

        auto debatcherOptionsPtr = DebatcherOptions::create(options);
        if (debatcherOptionsPtr) {
            pm.addPass(IE::createDebatcherPass(*debatcherOptionsPtr, log));
            pm.addPass(IE::createOutlinerPass(options, log));
            pm.addPass(IE::createDeDebatcherPass(*debatcherOptionsPtr, log));
            if (auto reorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(*debatcherOptionsPtr, log)) {
                pm.addPass(IE::createOverrideTileExecutorNumPass(*reorderingOptionsPtr, log));
            }
        } else {
            pm.addPass(IE::createOutlinerPass(options, log));
            pm.addPass(IE::createDuplicateFQAcrossFunctionCallsPass(log));
        }
    }
}

void vpux::IE::arch40xx::buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
                                                   Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createOptimizeUnalignedQDQSeqPass(log));
    pm.addPass(IE::createSwapFakeQuantWithReshapeAndStridedSlicePass(log));
    pm.addPass(IE::createSwapConvertWithReshapeKindOpsPass(log));
    if (options.enableAlignScales) {
        pm.addPass(IE::createAlignScalesPass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }
    if (options.enableAdjustNonZeroFakeQuant) {
        pm.addPass(IE::createAdjustNonZeroFakeQuantPass(log));
    }
    if (options.enableMatmulMixedPrecisionDecomposition) {
        pm.addPass(IE::createProcessAsymmetricZeroPointsForMatmulPass(
                options.matmulMixedPrecisionDecompositionRatio.getValue(), log));
    }

    pm.addPass(IE::createSplitFakeQuantPass(log));

    pm.addPass(mlir::createCanonicalizerPass());  // Note: folds constants before convert-to-dequantize
    pm.addPass(IE::createConvertToQuantizedOpsPass(log));
    if (options.enablePropagateQuantDequant) {
        pm.addPass(mlir::createCanonicalizerPass(grc));
        pm.addPass(
                IE::createPropagateAndFuseQuantizeDequantizePass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }
    pm.addPass(IE::createFuseConvertWithQDQPass(log));
    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    pm.addPass(IE::createPropagateDequantThroughConcatPass(log));
    pm.addPass(IE::createConvertWeightsToU8Pass(log));
    pm.addPass(IE::createFuseQuantizedOpsPass(
            /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
            /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));

    // Enable sequence FuseQuantizedOps->FuseActivationOps->FuseOutstandingDequant->ConvertToMixedPrecision->
    // ConvertQuantizeOpsToNceOps. The sequence allows Conv->Quantize->Dequantize->LeakyReLU->Quantize->Dequantize
    // fused into a single Conv.
    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));

    if (options.enableConvertWeightsToU8I4) {
        pm.addPass(IE::createConvertWeightsToI8Pass(log));
    }
    if (options.enableConvertToPalletizationLUT) {
        pm.addPass(IE::createConvertToPalletizationLUT(log));
    }
    pm.addPass(IE::createConvertToMixedPrecision(isOptionEnabled(options.enableFloatInQuantWeightsMixedMode), log));
    if (options.enableQuantDequantRemoval) {
        pm.addPass(IE::createRemoveQuantDequantSeqPass(log));
    }
    if (options.enableConvertWeightsToU8I4) {
        pm.addPass(IE::createConvertWeightsToU8Pass(log));
        pm.addPass(IE::createConvertWeightsToI4Pass(log));
    }
    // After the execution of ConvertWeightsToU8 could appear new cases when FuseQuantizedOps and
    // ConvertToMixedPrecision can be applied. The execution ConvertWeightsToU8 can align the data type of the operands
    // of NCE operations to U8, condition that is necessary in rewriters like: FuseWithEltwiseConverter,
    // FloatOutAddRewriter where it required that the operands to have the same data type and this happens only after
    // execution of ConvertWeightsToU8.
    pm.addPass(IE::createFuseQuantizedOpsPass(
            /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
            /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    if (options.enableFuseOutstandingDequant) {
        pm.addPass(IE::createFuseOutstandingDequant(log));

        // This is a short term solution to call ConvertToMixedPrecision when we have
        //     Original subgraph
        //         Conv -> FQ1 -> FQ2 -> Conv (FQ1 and FQ2 have different params)
        //     At this point
        //        (Conv-Q1-DQ1) -> Q2 -> (DQ2-Conv)
        // In long term need to consider a new pass to fuse FQs with different params
    }
    // Note: this ConvertToMixedPrecision call serves both FuseQuantizedOps and
    // FuseOutstandingDequant
    pm.addPass(IE::createConvertToMixedPrecision(isOptionEnabled(options.enableFloatInQuantWeightsMixedMode), log));

    if (options.enableFuseOutstandingQuant) {
        pm.addPass(IE::createFuseOutstandingQuantPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(options.runtimeDequantizationLimit,
                                             isOptionEnabled(options.enableRuntimeDequant), log));
    if (options.enableDynamicQuant) {
        pm.addPass(IE::createWeightsQuantFusedIntoTaskPass(log));
    }
    pm.addPass(IE::createOptimizePrecisionAcrossFunctionCallsPass(log));

    // E#176434: remove option
    if (options.enableConvertQuantizeOpsToNceOps) {
        pm.addPass(IE::createConvertQuantizeOpsToNceOpsPass(log));
    }

    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::arch40xx::buildFinalTransformationPipeline(mlir::OpPassManager& pm,
                                                          const IE::arch40xx::DefaultHWOptions& options, Logger log) {
    pm.addPass(IE::createAdaptODUPermutePass(log));
    // Operation Conversions
    if (options.enableConvertExpandToConvPass) {
        pm.addPass(IE::createConvertExpandToConvPass(log));
    }

    // Operation Fusions
    pm.addPass(IE::createOptimizeIdentityPoolPass(log));
    if (options.enableFuseD2SExpand) {
        pm.addPass(IE::createFuseD2SExpandChannelsPass(log));
    }

    // Operation optimizations
    pm.addPass(IE::createPropagateShapeCastPass(log));
    pm.addPass(IE::createPropagatePermuteCastThroughDequantizePass(log));
    pm.addPass(IE::createMoveDynamicDequantizeToUserPass(log));
}

//
// DefaultHWPipeline
//

void vpux::IE::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(locverif::createStartLocationVerifierPass(log, options.locationsVerificationMode));

    IE::arch40xx::buildOutliningPipeline(pm, options, log);

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, log);
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableReduceNumTilesForSmallModelsPass) {
        pm.addPass(IE::createReduceNumTilesForSmallModelsPass(log));
    }

    // Level 3 : Topology
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }

    if (options.enableFlashSDPAConversion) {
        pm.addPass(IE::createConvertSDPAToFlashSDPAPass(log));
    }

    if (options.enableDynamicShapeTransformationsPipeline) {
        IE::buildDynamicShapeTransformationsPipeline(pm, IE::DynamicShapeTransformOptions(options), log);
    }
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);

    if (options.enableAdjustPrecisionPipeline) {
        IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);
    }

    // Couldn't move the pass before convert_precision_to_fp16 because of regressions, extra conversions are added
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));
    IE::buildOperationConversionPipeline(pm, IE::OperationConversionOptions(options), log);

    IE::buildAdjustShapePipeline(pm, log);
    IE::buildSplitLargeOpsPipeline(pm, log);
    IE::buildConvertToEfficientOpsPipeline(pm, IE::ConvertToEfficientOpsOptions(options), log);

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(mlir::createCSEPass());

    IE::buildHandleHyperParametersPipeline(pm, IE::HyperParameterOptions(options), log);
    IE::buildConvertToConvolutionPipeline(pm, log);
    IE::buildReorderFakeQuantizePipeline(pm, IE::ReorderFakeQuantizeOptions(options), log);

    pm.addPass(locverif::createStopLocationVerifierPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    if (options.enableLowPrecision) {
        IE::arch40xx::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
        pm.addPass(IE::createSwapViewOpAndClampPass(log));
    }
    IE::buildOptimizeActivationsPipeline(pm, IE::OptimizeActivationsOptions(options), log);

    if (options.enableSEPtrsOperations && options.enableSplitBilinerIntoHAndW) {
        pm.addPass(IE::createSplitBilinerIntoHAndWPass(log));
    }

    if (options.enableBilinearInterpolateOnDPU) {
        pm.addPass(IE::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }

    IE::buildBatchTransformationPipeline(pm, BatchUnrollOptions::create(options, log), options.enableUpstreamSlice,
                                         log);

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    IE::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(pm, IE::ExpandActivationChannelsOptions(options),
                                                                   log);

    IE::buildOptimizeViewLikeOpsPipeline(pm, log);

    IE::buildOptimizeSliceOpPipeline(pm, log);

    IE::buildDimensionAlignmentPipeline(pm, IE::ExpandActivationChannelsOptions(options), log);

    IE::arch40xx::buildFinalTransformationPipeline(pm, options, log);

    // Shave related optimization
    pm.addPass(IE::createLoadExternalKernelResourcesPass(log));
    if (options.enableShaveCodeGen) {
        IE::buildShaveCodeGenPipeline(pm, log);
    }

    // Logging of optimizations post-check at the end of the pipeline
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }
}

void vpux::IE::arch40xx::buildReferenceSWPipeline(mlir::OpPassManager& pm,
                                                  const IE::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, log);

    // Level 3 : Topology

    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    // Couldn't move the pass before convert_precision_to_fp16 because of regressions, extra conversions are added
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    // Resolve group quant MatMul pattern
    pm.addPass(mlir::createCSEPass());
    pm.addPass(IE::createUniquifySimilarOpsPass(log));
    pm.addPass(IE::createMergeParallelFullyConnectedPass(log));
    pm.addPass(IE::createUnrollGroupQuantizePass(log));
    pm.addPass(IE::createUnrollFullyConnectedPass(log));
    pm.addPass(IE::createMergeFullyConnectedPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(options.runtimeDequantizationLimit,
                                             isOptionEnabled(options.enableRuntimeDequant), log));
    if (options.enableMergeFakeQuant) {
        pm.addPass(IE::createMergeFakeQuantPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableShaveCodeGen) {
        IE::buildShaveCodeGenPipeline(pm, log);
    }
}

//
// registerIEPipelines
//

void vpux::IE::arch40xx::registerIEPipelines() {
    mlir::PassPipelineRegistration<IE::arch40xx::DefaultHWOptions>(
            "default-hw-mode-ie", "IE dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options) {
                IE::arch40xx::buildDefaultHWPipeline(pm, options);
            });
    mlir::PassPipelineRegistration<IE::LowPrecisionTransformOptions>(
            "initial-low-precision-transformations",
            "[LEGALIZATION] Initial Low Precision Transformations, convert initial low precision IR operations to "
            "equivalent operations supported by the lower compilation levels",
            [](mlir::OpPassManager& pm, const IE::LowPrecisionTransformOptions& options) {
                IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<LowPrecisionOptions>(
            "low-precision", "[OPTIMIZATION] Low precision transformations",
            [](mlir::OpPassManager& pm, const LowPrecisionOptions& options) {
                IE::arch40xx::buildLowPrecisionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DynamicShapeTransformOptions>(
            "dynamic-shape-transformations", "[LEGALIZATION] Introduces operation to handle dynamic shapes",
            [](mlir::OpPassManager& pm, const DynamicShapeTransformOptions& options) {
                IE::buildDynamicShapeTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<IE::arch40xx::DefaultHWOptions>(
            "reference-sw-mode-ie", "IE dialect part of Reference SW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options) {
                IE::arch40xx::buildReferenceSWPipeline(pm, options);
            });
}
