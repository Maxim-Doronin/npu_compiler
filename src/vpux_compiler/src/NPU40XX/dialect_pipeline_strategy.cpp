//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/options_mapper.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"

using namespace vpux;

namespace {

//
// OptionsSetup40XX
//

class DefaultHWSetup40XX : public OptionsSetup<DefaultHWSetup40XX, DefaultHWOptions40XX> {
public:
    using Base = OptionsSetup<DefaultHWSetup40XX, DefaultHWOptions40XX>;
    using Base::Base;
    friend Base;

protected:
    // Note: must be static as we call ConcreteModel::setupOptionsImpl() from the ctor of base class
    static void setupOptionsImpl(DefaultHWOptions40XX& options, const intel_npu::Config& config) {
        setupPWLMCompilationParams(options.optimizationLevel, options, options.workloadManagementEnable);
        options.enableProfiling = config.get<intel_npu::PERF_COUNT>();
        options.enableConvertAvgPoolToDWConv = false;
        options.enableHandleAsymmetricStrides = false;
        // TODO: E#-108844 Support Compressed activation with Partial workload management
        if (options.workloadManagementEnable) {
            options.enableCompressActivationSpill = false;
        }
        options.updateBatchCompileOptionsFromString(config.get<intel_npu::BATCH_COMPILER_MODE_SETTINGS>());
    }
};

class ShaveCodeGenSetup40XX : public ShaveCodeGenSetupBase<DefaultHWOptions40XX> {
public:
    using Base = ShaveCodeGenSetupBase<DefaultHWOptions40XX>;
    using Base::Base;
};

class ReferenceSWSetup40XX : public ReferenceSwSetupBase<ReferenceSWOptions40XX> {
public:
    using Base = ReferenceSwSetupBase<ReferenceSWOptions40XX>;
    using Base::Base;
};

//
// DialectPipelineStrategy40XX: [DefaultHW, ShaveCodeGen]
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
                pm, _optionsContainer->getPipelineOptions().enableInPlaceBufferization, log);
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
        pm.addPass(IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                                    log));
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

        pm.addPass(VPU::createTilingStrategyAssignmentPass(/*enablePrefetchTiling=*/false, false, "true", log));
        pm.addPass(VPU::arch37xx::createApplyTilingMVN1SumPass(/*enablePrefetchTiling=*/false, log));
        pm.addPass(VPU::createApplyTilingPass(/*enableSCFTiling=*/false, log));

        pm.addPass(VPU::createComputeInterpolateCoordinatesPass(/*enableExplicitDistributionInfoAttr=*/true, log));

        pm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));

        // Lowering to VPUIP
        vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, options.enableInPlaceBufferization, log);

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

        pm.addPass(VPUIP::arch40xx::createAddStartBarrierPass(log));
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

std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(VPU::CompilationMode compilationMode,
                                                                                  const intel_npu::Config& config) {
    switch (compilationMode) {
    case VPU::CompilationMode::DefaultHW: {
        return std::make_unique<DialectPipelineStrategy40XX<DefaultHWSetup40XX>>(config);
    }
    case VPU::CompilationMode::ShaveCodeGen: {
        return std::make_unique<DialectPipelineStrategy40XX<ShaveCodeGenSetup40XX>>(config);
    }
    case VPU::CompilationMode::ReferenceSW: {
        return std::make_unique<DialectPipelineStrategy40XX<ReferenceSWSetup40XX>>(config);
    }
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

//
// createDialectPipelineStrategy40XX [lit-tests]
//

template <class OptionsType>
std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(
        const VPU::InitCompilerOptions* initCompilerOptions, const OptionsType* options) {
    auto wrapper = std::make_unique<OptionsWrapper<OptionsType>>(initCompilerOptions, options);
    return std::make_unique<DialectPipelineStrategy40XX<OptionsWrapper<OptionsType>>>(std::move(wrapper));
}

template std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(
        const VPU::InitCompilerOptions*, const DefaultHWOptions40XX*);
template std::unique_ptr<IDialectPipelineStrategy> vpux::createDialectPipelineStrategy40XX(
        const VPU::InitCompilerOptions*, const ReferenceSWOptions40XX*);
