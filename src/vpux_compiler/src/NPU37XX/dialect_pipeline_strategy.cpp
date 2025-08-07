//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU37XX/pipeline_options.hpp"

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/pipelines/options_setup.hpp"

using namespace vpux;

namespace {

//
// OptionsSetup37XX
//

class DefaultHWSetup37XX : public OptionsSetupBase<DefaultHWSetup37XX, DefaultHWOptions37XX> {
public:
    using Base = OptionsSetupBase<DefaultHWSetup37XX, DefaultHWOptions37XX>;
    using Base::Base;
};

class ShaveCodeGenSetup37XX : public OptionsSetupBase<ShaveCodeGenSetup37XX, DefaultHWOptions37XX> {
public:
    using Base = OptionsSetupBase<ShaveCodeGenSetup37XX, DefaultHWOptions37XX>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base::Base;

private:
    static void setupOptionsImpl(DefaultHWOptions37XX& options, const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, config);

        // E#154882 Ensure standard VPUX passes compatibility with ShaveCodeGen path
        overwriteIfUnset(options.locationsVerificationMode, "off");
        overwriteIfUnset(options.enableShaveKernelTiling, false);
        // E#154882 Ensure standard VPUX passes compatibility with ShaveCodeGen path
        overwriteIfUnset(options.enableOptimizeCopies, false);
    }
};

class ReferenceSWSetup37XX : public OptionsSetupBase<ReferenceSWSetup37XX, ReferenceSWOptions37XX> {
public:
    using Base = OptionsSetupBase<ReferenceSWSetup37XX, ReferenceSWOptions37XX>;
    using Base::Base;
};

class WSMonolithicSetup37XX final : public WSMonolithicSetupBase<WSMonolithicSetup37XX, DefaultHWOptions37XX> {
public:
    using Base = WSMonolithicSetupBase<WSMonolithicSetup37XX, DefaultHWOptions37XX>;
    using Base::Base;
};

//
// DialectPipelineStrategy37XX
//

template <class OptionsContainerType, class Enable = void>
class DialectPipelineStrategy37XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy37XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy37XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch37xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::arch37xx::buildLowerIE2VPUPipeline(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch37xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::arch37xx::buildLowerVPU2VPUIPPipeline(
                pm, _optionsContainer->getPipelineOptions().enableInPlaceBufferization,
                _optionsContainer->getPipelineOptions().useMemrefForHostFunctionBufferization, log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch37xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<OptionsContainerType> _optionsContainer;
};

//
// DialectPipelineStrategy37XX: [ReferenseSW]
// This implementation will be chosen if OptionsContainerType contains ReferenceSWOptions
//

template <typename T>
using Has37XXSWOptions = typename std::enable_if_t<std::is_same_v<typename T::value_type, ReferenceSWOptions37XX>>;

template <class OptionsContainerType>
class DialectPipelineStrategy37XX<OptionsContainerType, Has37XXSWOptions<OptionsContainerType>> final :
        public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy37XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy37XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildReferenceSWPipeline(mlir::OpPassManager& pm, Logger log) override {
        auto& options = _optionsContainer->getPipelineOptions();
        const auto grc = getDefaultGreedyRewriteConfig();

        // No passes should be run before this pipeline, with very few exceptions.
        IE::buildPostImportPipeline(pm, log);

        // Level 3 : Topology
        IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options),
                                                                      log);
        IE::arch37xx::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
        IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

        // Resolve group quant MatMul pattern
        pm.addPass(IE::createUniquifyOpsPass(log));
        pm.addPass(IE::createMergeParallelFullyConnectedPass(log));
        pm.addPass(IE::createUnrollGroupQuantizePass(log));
        pm.addPass(IE::createUnrollFullyConnectedPass(log));
        if (options.fuseScalesToAccumulate) {
            pm.addPass(IE::createFuseScalesToAccumulatePass(log));
        }
        pm.addPass(IE::createConvertMatMulToConvPass(log));
        if (options.enableConvertFCToConv) {
            pm.addPass(IE::createConvertFCToConvPass(log));
        }

        pm.addPass(IE::createResolveStridedSlicePass(log));
        pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
        pm.addPass(IE::createConvertNceOpsTo4DPass(log));
        pm.addPass(IE::createConvertShapeTo4DPass(log));
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

        IE::arch37xx::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
        pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

        pm.addPass(IE::createConvertToMemPermutePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        // Lowering to VPU
        pm.addPass(createConvertLayers2VPUPass(log));
        pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
        pm.addPass(VPU::arch37xx::createSplitRealDFTOpsPass(log));
        pm.addPass(VPU::createSplitGRUSequencePass(log));
        pm.addPass(VPU::arch37xx::createDecomposeMVNPass(log));
        pm.addPass(VPU::createAddSwOpAuxiliaryBufferPass(log));

        pm.addPass(VPU::createTilingStrategyAssignmentPass(
                /*enablePrefetchTiling=*/false, /*enableVPUNNCostForTiling*/ false,
                /*enableShaveDDRAccessOptimization*/ "true", log));
        pm.addPass(VPU::arch37xx::createApplyTilingMVN1SumPass(/*enablePrefetchTiling=*/false, log));
        pm.addPass(VPU::createApplyTilingPass(/*enableSCFTiling=*/false, log));
        pm.addPass(VPU::createComputeInterpolateCoordinatesPass(/*enableExplicitDistributionInfoAttr*/ false, log));

        pm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));

        // Lowering to VPUIP
        vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, options.enableInPlaceBufferization,
                                                    /*useMemrefForHostFunctionBufferization*/ false, log);

        // Level 2 : Abstract RunTime

        pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));

        pm.addPass(VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(log));

        pm.addPass(VPUIP::createCopyOpTilingPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableProfiling && options.enableSWProfiling) {
            pm.addPass(VPUIP::createActShaveProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
        }

        pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));

        pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

        VPUIP::buildAsyncSchedulingPipeline(pm, log);

        if (options.enableSWKernelPrefetchingReserveMem) {
            pm.addPass(VPUIP::createSWKernelPrefetchingReserveMemPass(log));
        }

        pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
        pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
        pm.addPass(VPUIP::createLinearizationPass(log));
        pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

        pm.addPass(VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(log));

        VPUIP::buildHardwareAdaptationPipeline(pm, log);

        pm.addPass(VPURT::arch37xx::createAddFinalBarrierPass(log));

        // Level 1 : VPU RunTime

        if (options.enableProfiling) {
            pm.addPass(VPUIP::createCaptureWorkpointPass(log));
            pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
            pm.addPass(Core::createMoveDeclarationsToTopPass(log));
        }

        pm.addPass(VPURT::createAssignPhysicalBarriersPass(options.enableColorBinPhysicalBarrierAssignment,
                                                           std::nullopt, std::nullopt, log));
        pm.addPass(VPURT::createBarrierSimulationPass(log));
        pm.addPass(VPUIP::createUpdateSwKernelParamsPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        pm.addPass(Const::createConstantFoldingPass());
        pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
    }

private:
    std::unique_ptr<OptionsContainerType> _optionsContainer;
};

}  // namespace

//
// createDialectPipelineStrategy37XX
//

std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy37XX(
        config::CompilationMode compilationMode, const intel_npu::Config& config) {
    switch (compilationMode) {
    case config::CompilationMode::DefaultHW: {
        return std::make_unique<DialectPipelineStrategy37XX<DefaultHWSetup37XX>>(config);
    }
    case config::CompilationMode::ShaveCodeGen: {
        return std::make_unique<DialectPipelineStrategy37XX<ShaveCodeGenSetup37XX>>(config);
    }
    case config::CompilationMode::ReferenceSW: {
        return std::make_unique<DialectPipelineStrategy37XX<ReferenceSWSetup37XX>>(config);
    }
    case config::CompilationMode::WSMonolithic: {
        return std::make_unique<DialectPipelineStrategy37XX<WSMonolithicSetup37XX>>(config);
    }
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

//
// createDialectPipelineStrategy37XX [lit-tests]
//

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy37XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions37XX* options) {
    auto wrapper = std::make_unique<DefaultHWSetup37XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy37XX<DefaultHWSetup37XX>>(std::move(wrapper));
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy37XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const ReferenceSWOptions37XX* options) {
    auto wrapper = std::make_unique<ReferenceSWSetup37XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy37XX<ReferenceSWSetup37XX>>(std::move(wrapper));
}
