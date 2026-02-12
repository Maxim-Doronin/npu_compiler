//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/locverif/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::IE::arch50xx::buildInitialLowPrecisionTransformationsPipeline(mlir::OpPassManager& pm,
                                                                         const LowPrecisionTransformOptions& options,
                                                                         Logger log) {
    pm.addPass(IE::createReshapeMatMulInputsPass(options.enableGroupedMatMul, log));
    pm.addPass(IE::createConvertScalarToTensorPass(log));
    pm.addPass(IE::createQDQOptimizationAggressivePass(options.fuseFQAndMulWithNonConstInput, log));
    pm.addPass(IE::createConsolidateNF4WeightsPatternPass(log));
    pm.addPass(IE::createInitialLowPrecisionTransformationsPipelineRewriterExecutorPass(log));
    pm.addPass(IE::createFuseInputScaleShiftPass(log));
    pm.addPass(IE::arch50xx::createConvertFakeConvertToFakeQuantizePass(log));
    pm.addPass(IE::createConvertMinMaxToClampPass(log));
    pm.addPass(IE::createFoldActivationBeforeFQPass(log));
}

void vpux::IE::arch50xx::buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
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
        pm.addPass(IE::createProcessAsymmetricZeroPointsForMatmulPass(options.matmulMixedPrecisionDecompositionRatio,
                                                                      log));
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
    pm.addPass(IE::createOptimizePrecisionAcrossFunctionCallsPass(log));

    // E#176434: remove option
    if (options.enableConvertQuantizeOpsToNceOps) {
        pm.addPass(IE::createConvertQuantizeOpsToNceOpsPass(log));
    }

    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::arch50xx::buildConvertToEfficientOpsPipeline(mlir::OpPassManager& pm,
                                                            const ConvertToEfficientOpsOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertDivideToMultiplyPass(log));
    // NOTE: ReassociateMultiply relies on ConvertDivideToMultiply
    pm.addPass(IE::createReassociateMultiplyPass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    // NOTE: Canonicalizer required after ConvertShapeTo4DPass
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertGatherElementsToGatherPass(log));
    pm.addPass(IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createConvertSplitConcatToAffineReshapePass(log));
    pm.addPass(IE::createConvertBranchesConcatToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createSwapPadLayerPass(log));
    // NOTE: apply FuseStaticScale after ConvertDivideToMultiply to increase
    // the applicability
    pm.addPass(IE::createFuseStaticScalePass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createBroadcastInputForAddPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    // E#79878: Solve eltwise single layer test failure.
    // SwapOperations pass may generate non-4D AddOp.
    // If AddOp appears here means that it cannot be fused into NCE task.
    // So convert it's shape to 4D and then convert this AddOp to ScaleShift.
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    // NOTE: Canonicalizer required after ConvertShapeTo4DPass
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    // NOTE: Required to avoid performance regression related to AddOp
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    if (options.enableD2SToTransposedConvConversion) {
        pm.addPass(
                IE::createConvertDepth2SpaceToTransposedConvPass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }
    // NOTE: SwapD2SAndScaleShift depends on ConvertDepth2SpaceToTransposedConv
    pm.addPass(IE::createSwapD2SAndScaleShiftPass(log));
    pm.addPass(IE::createConvertReverseToDWConvPass(log));
    if (options.enableConvertDeformableConvToConv) {
        pm.addPass(IE::createConvertDeformableConvToConvPass(log));
    }
}

void vpux::IE::arch50xx::buildFinalTransformationPipeline(mlir::OpPassManager& pm,
                                                          const IE::arch50xx::DefaultHWOptions& options, Logger log) {
    pm.addPass(IE::createAdaptODUPermutePass(log));
    if (options.broadcastInputForMultiply) {
        pm.addPass(IE::createBroadcastInputForMultiplyPass(log));
        pm.addPass(IE::createConvertBroadcastToTilePass(log));
    }
    if (options.enableConvertExpandToConvPass) {
        pm.addPass(IE::createConvertExpandToConvPass(log));
    }

    // Operation optimizations
    pm.addPass(IE::createPropagateShapeCastPass(log));
    pm.addPass(IE::createPropagatePermuteCastThroughDequantizePass(log));
    pm.addPass(IE::createMoveDynamicDequantizeToUserPass(log));

    // Operation Fusions
    pm.addPass(IE::createOptimizeIdentityPoolPass(log));
    pm.addPass(IE::createFuseSoftMaxConvertPass(log));
    pm.addPass(IE::createFuseLogSoftmaxVariantsPass(log));
    if (options.enableFuseD2SExpand) {
        pm.addPass(IE::createFuseD2SExpandChannelsPass(log));
    }
}

//
// DefaultHWPipeline
//

void vpux::IE::arch50xx::buildDefaultHWPipeline(mlir::OpPassManager& pm, const IE::arch50xx::DefaultHWOptions& options,
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
    IE::arch50xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
    if (options.enableAdjustPrecisionPipeline) {
        IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);
    }

    // Couldn't move the pass before convert_precision_to_fp16 because of regressions, extra conversions are added
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    IE::buildOperationConversionPipeline(pm, IE::OperationConversionOptions(options), log);

    IE::buildAdjustShapePipeline(pm, log);
    IE::buildSplitLargeOpsPipeline(pm, log);
    IE::arch50xx::buildConvertToEfficientOpsPipeline(pm, IE::ConvertToEfficientOpsOptions(options), log);

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);
    pm.addPass(mlir::createCSEPass());

    IE::buildHandleHyperParametersPipeline(pm, IE::HyperParameterOptions(options), log);
    IE::buildConvertToConvolutionPipeline(pm, log);
    IE::buildReorderFakeQuantizePipeline(pm, IE::ReorderFakeQuantizeOptions(options), log);

    // All locations unique with full verification after each pass to this point
    pm.addPass(locverif::createStopLocationVerifierPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    if (options.enableLowPrecision) {
        IE::arch50xx::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
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

    IE::arch50xx::buildFinalTransformationPipeline(pm, options, log);

    // Shave related optimization
    pm.addPass(IE::createLoadExternalKernelResourcesPass(log));
    if (options.enableShaveCodeGen) {
        IE::buildShaveCodeGenPipeline(pm, log);
    }

    // Logging of optimizations at the end of the pipeline
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }
}

void vpux::IE::arch50xx::buildReferenceSWPipeline(mlir::OpPassManager& pm,
                                                  const IE::arch50xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, log);

    // Level 3 : Topology

    IE::arch50xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
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

void vpux::IE::arch50xx::registerIEPipelines() {
    mlir::PassPipelineRegistration<IE::arch50xx::DefaultHWOptions>(
            "default-hw-mode-ie", "IE dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch50xx::DefaultHWOptions& options) {
                IE::arch50xx::buildDefaultHWPipeline(pm, options);
            });
    mlir::PassPipelineRegistration<IE::LowPrecisionTransformOptions>(
            "initial-low-precision-transformations",
            "[LEGALIZATION] Initial Low Precision Transformations, convert initial low precision IR operations to "
            "equivalent operations supported by the lower compilation levels",
            [](mlir::OpPassManager& pm, const IE::LowPrecisionTransformOptions& options) {
                IE::arch50xx::buildInitialLowPrecisionTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<LowPrecisionOptions>(
            "low-precision", "[OPTIMIZATION] Low precision transformations",
            [](mlir::OpPassManager& pm, const LowPrecisionOptions& options) {
                IE::arch50xx::buildLowPrecisionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DynamicShapeTransformOptions>(
            "dynamic-shape-transformations", "[LEGALIZATION] Introduces operation to handle dynamic shapes",
            [](mlir::OpPassManager& pm, const DynamicShapeTransformOptions& options) {
                IE::buildDynamicShapeTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<IE::arch50xx::DefaultHWOptions>(
            "reference-sw-mode-ie", "IE dialect part of Reference SW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch50xx::DefaultHWOptions& options) {
                IE::arch50xx::buildReferenceSWPipeline(pm, options);
            });
}
