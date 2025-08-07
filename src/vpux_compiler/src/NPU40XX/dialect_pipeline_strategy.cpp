//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/pipelines/options_setup.hpp"

using namespace vpux;

namespace {

//
// OptionsSetup40XX
//

class DefaultHWSetup40XX final : public OptionsSetupBase<DefaultHWSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<DefaultHWSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base::Base;

    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options) {
        Base::setupLitTestOptionsImpl(options);
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, config);
        if (config.get<intel_npu::TURBO>()) {
            overwriteIfUnset(options.optimizationLevel, 3);
        }
        setupOptionsCommon(options);
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options) {
        setupParamsAccordingToOptimizationLevel(options.optimizationLevel, options, options.workloadManagementEnable);
        setupPWLMParams(options);
    }
};

class ShaveCodeGenSetup40XX : public OptionsSetupBase<ShaveCodeGenSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<ShaveCodeGenSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
};

class ReferenceSWSetup40XX : public OptionsSetupBase<ReferenceSWSetup40XX, ReferenceSWOptions40XX> {
public:
    using Base = OptionsSetupBase<ReferenceSWSetup40XX, ReferenceSWOptions40XX>;
    using Base::Base;
};

class HostCompileSetup40XX : public OptionsSetupBase<HostCompileSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetupBase<HostCompileSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options) {
        // DefaultHW options
        DefaultHWSetup40XX::setupLitTestOptionsImpl(options);

        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, const intel_npu::Config& config) {
        // DefaultHW options
        DefaultHWSetup40XX::setupOptionsImpl(options, config);

        // HostCompileSetup40XX common options
        setupOptionsCommon(options);
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options) {
        // DefaultHW specific options
        DefaultHWSetup40XX::setupOptionsCommon(options);

        // HostCompile specific options
        overwriteIfUnset(options.enableDynamicShapeTransformationsPipeline, false);
        overwriteIfUnset(options.enableSCFTiling, true);
        overwriteIfUnset(options.enableScfComputeOpsOutlining, true);
        overwriteIfUnset(options.useMemrefForHostFunctionBufferization, true);
    }
};
class WSMonolithicSetup40XX final : public WSMonolithicSetupBase<WSMonolithicSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = WSMonolithicSetupBase<WSMonolithicSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options) {
        Base::setupLitTestOptionsImpl(options);
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, config);
        setupOptionsCommon(options);
    }

    static void setupOptionsCommon(DefaultHWOptions40XX& options) {
        setupParamsAccordingToOptimizationLevel(options.optimizationLevel, options, options.workloadManagementEnable);
        setupPWLMParams(options);
    }
};

class WSInitSetup40XX : public WSInitSetupBase<WSInitSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = WSInitSetupBase<WSInitSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    friend Base::Base;

private:
    static void setupLitTestOptionsImpl(DefaultHWOptions40XX& options) {
        Base::setupLitTestOptionsImpl(options);
        setupPWLMParams(options);
    }

    static void setupOptionsImpl(DefaultHWOptions40XX& options, const intel_npu::Config& config) {
        Base::setupOptionsImpl(options, config);
        setupPWLMParams(options);
    }
};

//
// DialectPipelineStrategy40XX
//

template <class OptionsContainerType, class Enable = void>
class DialectPipelineStrategy40XX final : public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy40XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy40XX(std::unique_ptr<OptionsContainerType> optionsContainer)
            : _optionsContainer(std::move(optionsContainer)) {
    }

    void initializePipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::buildInitCompilerPipeline(pm, _optionsContainer->getInitCompilerOptions(), log.nest());
    }

    void buildIEPipeline(mlir::OpPassManager& pm, Logger log) override {
        IE::arch40xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        // Lowering to VPU
        if (_optionsContainer->getPipelineOptions().enableM2I) {
            pm.addPass(createConvertIEToVPUM2IPass(log));
        }

        vpux::arch37xx::buildLowerIE2VPUPipeline(pm, log);
    }

    void buildVPUPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPU::arch40xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

    void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        vpux::arch37xx::buildLowerVPU2VPUIPPipeline(
                pm, _optionsContainer->getPipelineOptions().enableInPlaceBufferization,
                _optionsContainer->getPipelineOptions().useMemrefForHostFunctionBufferization, log);
    }

    void buildVPUIPPipeline(mlir::OpPassManager& pm, Logger log) override {
        VPUIP::arch40xx::buildDefaultHWPipeline(pm, _optionsContainer->getPipelineOptions(), log);
    }

private:
    std::unique_ptr<OptionsContainerType> _optionsContainer;
};

//
// DialectPipelineStrategy40XX: [ReferenseSW]
// This implementation will be chosen if OptionsContainerType contains ReferenceSWOptions
//

template <typename T>
using Has40XXSWOption = typename std::enable_if_t<std::is_same_v<typename T::value_type, ReferenceSWOptions40XX>>;

template <class OptionsContainerType>
class DialectPipelineStrategy40XX<OptionsContainerType, Has40XXSWOption<OptionsContainerType>> final :
        public IDialectPipelineStrategy {
public:
    explicit DialectPipelineStrategy40XX(const intel_npu::Config& config)
            : _optionsContainer(std::make_unique<OptionsContainerType>(config)) {
    }

    explicit DialectPipelineStrategy40XX(std::unique_ptr<OptionsContainerType> optionsContainer)
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
        pm.addPass(IE::createMergeFullyConnectedPass(log));
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
        pm.addPass(VPU::createAddSwOpAuxiliaryBufferPass(log));
        pm.addPass(VPU::createSplitGRUSequencePass(log));
        pm.addPass(VPU::arch37xx::createDecomposeMVNPass(log));

        pm.addPass(VPU::createTilingStrategyAssignmentPass(
                /*enablePrefetchTiling=*/false, /*enableVPUNNCostForTiling*/ false,
                /*enableShaveDDRAccessOptimization*/ "true", log));
        pm.addPass(VPU::arch37xx::createApplyTilingMVN1SumPass(/*enablePrefetchTiling=*/false, log));
        pm.addPass(VPU::createApplyTilingPass(/*enableSCFTiling=*/false, log));

        pm.addPass(VPU::createComputeInterpolateCoordinatesPass(/*enableExplicitDistributionInfoAttr=*/true, log));

        pm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));

        // Lowering to VPUIP
        vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, options.enableInPlaceBufferization,
                                                    options.useMemrefForHostFunctionBufferization, log);

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

        pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(DMAProfilingMode::SCRATCH, log));

        if (options.enableSWKernelPrefetchingReserveMem) {
            pm.addPass(VPUIP::createSWKernelPrefetchingReserveMemPass(log));
        }

        pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
        pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
        pm.addPass(VPUIP::createLinearizationPass(log));
        pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

        pm.addPass(VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(log));

        VPUIP::buildHardwareAdaptationPipeline(pm, log);

        pm.addPass(VPUIP::arch40xx::createAddStartBarrierPass(/*compilerBarrierProgramming=*/false, log));
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
    }

private:
    std::unique_ptr<OptionsContainerType> _optionsContainer;
};

}  // namespace

//
// createDialectPipelineStrategy40XX
//

std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(
        config::CompilationMode compilationMode, const intel_npu::Config& config) {
    switch (compilationMode) {
    case config::CompilationMode::DefaultHW: {
        return std::make_unique<DialectPipelineStrategy40XX<DefaultHWSetup40XX>>(config);
    }
    case config::CompilationMode::ShaveCodeGen: {
        return std::make_unique<DialectPipelineStrategy40XX<ShaveCodeGenSetup40XX>>(config);
    }
    case config::CompilationMode::ReferenceSW: {
        return std::make_unique<DialectPipelineStrategy40XX<ReferenceSWSetup40XX>>(config);
    }
    case config::CompilationMode::HostCompile: {
        return std::make_unique<DialectPipelineStrategy40XX<HostCompileSetup40XX>>(config);
    }
    case config::CompilationMode::WSMonolithic: {
        return std::make_unique<DialectPipelineStrategy40XX<WSMonolithicSetup40XX>>(config);
    }
    case config::CompilationMode::WSInit: {
        return std::make_unique<DialectPipelineStrategy40XX<WSInitSetup40XX>>(config);
    }
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

//
// createDialectPipelineStrategy40XX [lit-tests]
//

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const DefaultHWOptions40XX* options) {
    auto wrapper = std::make_unique<DefaultHWSetup40XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy40XX<DefaultHWSetup40XX>>(std::move(wrapper));
}

template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const ReferenceSWOptions40XX* options) {
    auto wrapper = std::make_unique<ReferenceSWSetup40XX>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy40XX<ReferenceSWSetup40XX>>(std::move(wrapper));
}

/// The reason this method is separate from the default and reference compilation modes is that it has to *copy* the
/// options in order to override them.
template <>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XXWS(
        config::CompilationMode compilationMode, const VPU::InitCompilerOptions* initCompilerOptions,
        const DefaultHWOptions40XX* options) {
    switch (compilationMode) {
    case config::CompilationMode::WSMonolithic: {
        auto wrapper = std::make_unique<WSMonolithicSetup40XX>(initCompilerOptions, options);
        return std::make_unique<DialectPipelineStrategy40XX<WSMonolithicSetup40XX>>(std::move(wrapper));
    }
    case config::CompilationMode::WSInit: {
        auto wrapper = std::make_unique<WSInitSetup40XX>(initCompilerOptions, options);
        return std::make_unique<DialectPipelineStrategy40XX<WSInitSetup40XX>>(std::move(wrapper));
    }
    default:
        VPUX_THROW("Unsupported compilation mode {0} for Monolithic WS.", config::stringifyEnum(compilationMode));
        return {};
    }
}
