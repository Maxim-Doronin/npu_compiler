//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/backend_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

#include "vpux/compiler/NPU40XX/conversion.hpp"

#include "vpux/compiler/options_mapper.hpp"
#include "vpux/utils/IE/config.hpp"

#include "intel_npu/config/options.hpp"

using namespace vpux;

//
// BackendPipelineStrategy40XX::buildELFPipeline
//

void BackendPipelineStrategy40XX::buildELFPipeline(mlir::PassManager& pm, const intel_npu::Config& config,
                                                   mlir::TimingScope& rootTiming, Logger log, bool useWlm) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    auto dpuDryRunMode = VPU::DPUDryRunMode::NONE;
    const auto compilationMode = getCompilationMode(config);
    auto backendCompilationOptions =
            BackendCompilationOptions40XX::createFromString(config.get<intel_npu::BACKEND_COMPILATION_PARAMS>());

    VPUX_THROW_UNLESS(backendCompilationOptions != nullptr,
                      "build ELF pipeline failed to parse BACKEND_COMPILATION_PARAMS: {0}",
                      config.get<intel_npu::BACKEND_COMPILATION_PARAMS>());

    if (compilationMode == VPU::CompilationMode::DefaultHW) {
        auto options = DefaultHWOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "build ELF pipeline failed to parse COMPILATION_MODE_PARAMS: {0}",
                          config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        setupPWLMCompilationParams(options->optimizationLevel, *options, useWlm);
        dpuDryRunMode = VPU::getDPUDryRunMode(options->dpuDryRun);
        backendCompilationOptions->enableDMAProfiling = options->enableDMAProfiling.getValue();
        backendCompilationOptions->enableShaveDDRAccessOptimization = options->enableShaveDDRAccessOptimization;
        backendCompilationOptions->enableDumpStatisticsOfWlmOps = options->enableDumpTaskStats;
        backendCompilationOptions->workloadManagementBarrierCountThreshold =
                options->workloadManagementBarrierCountThreshold;
        backendCompilationOptions->workloadManagementMode = options->workloadManagementMode;
        backendCompilationOptions->workloadManagementEnable = options->workloadManagementEnable;
        backendCompilationOptions->workloadManagementBarrierProgrammingMode =
                options->workloadManagementBarrierProgrammingMode.hasValue()
                        ? options->workloadManagementBarrierProgrammingMode
                        : WorkloadManagementBarrierProgrammingMode::UNKNOWN;

        if (!options->workloadManagementBarrierProgrammingMode.hasValue()) {
            switch (backendCompilationOptions->workloadManagementMode) {
            case WorkloadManagementMode::PWLM_V0_LCA:
                backendCompilationOptions->workloadManagementBarrierProgrammingMode =
                        WorkloadManagementBarrierProgrammingMode::LEGACY;
                break;
            case WorkloadManagementMode::PWLM_V1_BARRIER_FIFO:
                backendCompilationOptions->workloadManagementBarrierProgrammingMode =
                        WorkloadManagementBarrierProgrammingMode::NO_BARRIER_DMAS_SCHEDULED;
                break;
            case WorkloadManagementMode::PWLM_V2_PAGES:
                backendCompilationOptions->workloadManagementBarrierProgrammingMode =
                        WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED;
                break;
            }
        }
        backendCompilationOptions->workloadManagementDmaFifoType = options->workloadManagementDmaFifoType;
    }
    arch40xx::buildLowerVPUIP2ELFPipeline(pm, *backendCompilationOptions, log.nest(), dpuDryRunMode);
}
