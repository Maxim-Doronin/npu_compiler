//
// Copyright (C) 2023-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

#include <npu_40xx_nnrt.hpp>
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// buildLowerVPUIP2ELFPipeline
//

void vpux::arch40xx::buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                                 const BackendCompilationOptions40XX& backendCompilationOptions,
                                                 Logger log, VPU::DPUDryRunMode dpuDryRunMode) {
    log.info("BackendCompilationOptions:\n"
             "  workloadManagementEnable = {0}\n"
             "  workloadManagementMode = {1}\n"
             "  workloadManagementBarrierCountThreshold = {2}\n"
             "  workloadManagementBarrierProgrammingMode = {3}\n"
             "  workloadManagementDmaFifoType = {4}\n"
             "  enableMemorySideCache = {5}\n"
             "  enableDMAProfiling = {6}\n"
             "  enableShaveDDRAccessOptimization = {7}\n",
             backendCompilationOptions.workloadManagementEnable,
             stringifyEnum(backendCompilationOptions.workloadManagementMode),
             backendCompilationOptions.workloadManagementBarrierCountThreshold,
             stringifyEnum(backendCompilationOptions.workloadManagementBarrierProgrammingMode),
             stringifyEnum(backendCompilationOptions.workloadManagementDmaFifoType),
             backendCompilationOptions.enableMemorySideCache, backendCompilationOptions.enableDMAProfiling,
             backendCompilationOptions.enableShaveDDRAccessOptimization);

    pm.addPass(VPUMI40XX::createAddPlatformInfoPass(log));

    // Below VPURT WLM passes are placed in LowerVPUIP2ELF pipeline to leverage BarrierInfo
    // working on VPURT dialect and order barriers in a way suitable for WLM
    // Those can be moved to the end of VPURT once WLM rollback does not happen.
    // Currently only ELF backend is retriggered during rollback and IR after VPURT
    // needs to be left in a state suitable for nonWLM flow.
    if (backendCompilationOptions.workloadManagementEnable &&
        backendCompilationOptions.workloadManagementMode != WorkloadManagementMode::FWLM_V1_PAGES &&
        backendCompilationOptions.workloadManagementMode != WorkloadManagementMode::PWLM_V0_1_PAGES) {
        if (backendCompilationOptions.workloadManagementMode > WorkloadManagementMode::PWLM_V0_LCA) {
            pm.addPass(VPURT::createFindWlmEnqueueBarrierPass(
                    backendCompilationOptions.workloadManagementMode,
                    backendCompilationOptions.workloadManagementDmaFifoType == DMAFifoType::HW, log));
        } else {
            pm.addPass(VPURT::createOrderBarriersForWlmPass(log));
        }
    }

    pm.addPass(createConvertVPUIP2VPUMI40XXPass(log, backendCompilationOptions.enableMemorySideCache,
                                                backendCompilationOptions.allocateShaveStackFrames));
    pm.addPass(VPUMI40XX::createSetupProfilingVPUMI40XXPass(backendCompilationOptions.enableDMAProfiling, log));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(ELF::createAddABIVersionPass(log));
    elfSubsetPipelineVPUMI(pm, backendCompilationOptions.workloadManagementEnable,
                           backendCompilationOptions.workloadManagementMode,
                           backendCompilationOptions.enableDumpStatisticsOfWlmOps,
                           backendCompilationOptions.workloadManagementBarrierProgrammingMode, log);

    // To support forward compatibility between UD2024.44 and API version 11.4.10,
    // compiler by default set previous API version (11.4.10)
    pm.addPass(VPUMI40XX::createAddMappedInferenceVersionOpPass(log, npu40xx::NNRT_API_UD2024_44_MAJOR_VERSION,
                                                                npu40xx::NNRT_API_UD2024_44_MINOR_VERSION,
                                                                npu40xx::NNRT_API_UD2024_44_PATCH_VERSION));

    elfSubsetPipelineVPUASM(pm, backendCompilationOptions.workloadManagementEnable,
                            backendCompilationOptions.workloadManagementDmaFifoType == DMAFifoType::HW, log);

    pm.addPass(VPUIPDPU::createExpandDPUConfigPass(log));
    pm.addPass(ELF::createUpdateELFSectionFlagsPass(log, backendCompilationOptions.enableShaveDDRAccessOptimization));
    pm.addPass(createConvertVPUASM2NPUReg40XXPass(log, backendCompilationOptions.modelIdentifier));
    pm.addPass(createConvertVPUIPDPU2NPUReg40XXPass(log, dpuDryRunMode));

    pm.addPass(VPURegMapped::createDeduceDynamicMappedInferenceVersionPass(log));

    pm.addPass(ELF::createHandleAlignmentRequirementsPass(log));
    pm.addPass(ELF::createSetOpOffsetsPass(log));

    pm.addPass(ELF::createSetCMXSymbolValuePass(
            log, npu40xx::nn_public::VPU_WORKSPACE_ADDR, npu40xx::nn_public::VPU_WORKSPACE_SIZE,
            npu40xx::VPU_METADATA_STORAGE_START, npu40xx::nn_public::VPU_METADATA_SIZE));
    pm.addPass(ELF::createAddRelocationsForDynamicStridesDMAs(log));
    pm.addPass(ELF::createAddELFRelocationsPass(log));
    pm.addPass(ELF::createRemoveEmptyELFSectionsPass(log));
}

//
// buildElfSubsetPipelineVPUMI
//

void vpux::arch40xx::elfSubsetPipelineVPUMI(
        mlir::OpPassManager& pm, bool workloadManagementEnable, WorkloadManagementMode workloadManagementMode,
        bool enableDumpStatisticsOfWlmOps,
        WorkloadManagementBarrierProgrammingMode workloadManagementBarrierProgrammingMode, const Logger& log) {
    if (!workloadManagementEnable) {
        pm.addPass(VPUMI40XX::createBarrierComputationPass(log));
        pm.addPass(VPUMI40XX::createLinkAllOpsPass(log));
        pm.addPass(VPUASM::createHoistInputOutputsPass(log));
        pm.addPass(VPUMI40XX::createResolveTaskLocationPass(log));
        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));
    } else {
        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));

        if (workloadManagementMode != WorkloadManagementMode::FWLM_V1_PAGES &&
            workloadManagementMode != WorkloadManagementMode::PWLM_V0_1_PAGES) {
            // Generate barrier dependencies in IR if WLM mode requires it
            pm.addPass(VPUMI40XX::createBarrierTopologicalMappingPass(log));
        }

        pm.addPass(VPUMI40XX::createGroupExecutionOpsPass(log));
        if (workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES ||
            workloadManagementMode == WorkloadManagementMode::PWLM_V0_1_PAGES) {
            pm.addPass(VPUMI40XX::createConvertFetchDmasToFetchTaskOpsPass(log));
        } else {
            pm.addPass(VPUMI40XX::createAddFetchOpsPass(log));
        }

        pm.addPass(VPUMI40XX::createResolveWLMTaskLocationPass(log));
        pm.addPass(VPUMI40XX::createUnGroupExecutionOpsPass(log));
        pm.addPass(VPUMI40XX::createPropagateFinalBarrierPass(log));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(VPUMI40XX::createNextSameIdAssignmentPass(log));

        if (workloadManagementMode == WorkloadManagementMode::PWLM_V0_1_PAGES) {
            pm.addPass(VPUMI40XX::createUnrollFetchTaskOpsPass(log));
        }

        if (workloadManagementMode != WorkloadManagementMode::FWLM_V1_PAGES) {
            pm.addPass(VPUMI40XX::createAddEnqueueOpsPass(workloadManagementMode, log));
        }

        if (workloadManagementMode != WorkloadManagementMode::PWLM_V0_1_PAGES) {
            pm.addPass(VPUMI40XX::createUnrollFetchTaskOpsPass(log));
        }

        if (workloadManagementMode > WorkloadManagementMode::PWLM_V0_1_PAGES) {
            pm.addPass(VPUMI40XX::createAddBarrierConfigurationOps(workloadManagementMode,
                                                                   workloadManagementBarrierProgrammingMode, log));
        }

        if (workloadManagementMode != WorkloadManagementMode::FWLM_V1_PAGES) {
            pm.addPass(VPUMI40XX::createAddBootstrapBarriersPass(log));
        }

        pm.addPass(VPUMI40XX::createAddBootstrapWorkItemsPass(workloadManagementMode, log));

        if (workloadManagementMode != WorkloadManagementMode::FWLM_V1_PAGES) {
            pm.addPass(VPUMI40XX::createSplitEnqueueOpsPass(log));
        }

        pm.addPass(VPUMI40XX::createLinkEnqueueTargetsPass(workloadManagementMode, log));
        if (workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
            pm.addPass(VPUMI40XX::createUpdateEnqueueDMAInputAndOutput(log));
        }
        if (workloadManagementMode != WorkloadManagementMode::FWLM_V1_PAGES) {
            pm.addPass(VPUMI40XX::createUnrollEnqueueOpsPass(log));
        }

        if (workloadManagementMode > WorkloadManagementMode::PWLM_V0_1_PAGES) {
            pm.addPass(VPUMI40XX::createLinkEnqueueOpsForSameBarrierPass(log));
        }
        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));

        if (enableDumpStatisticsOfWlmOps) {
            pm.addPass(VPUMI40XX::createDumpStatisticsOfWlmOpsPass(log));
        }

        pm.addPass(VPUASM::createHoistInputOutputsPass(log));
    }
}

//
// buildElfSubsetPipelineVPUASM
//

void vpux::arch40xx::elfSubsetPipelineVPUASM(mlir::OpPassManager& pm, bool workloadManagementEnable,
                                             bool disableDmaSwFifo, const Logger& log) {
    pm.addPass(createConvertVPUMI40XX2VPUASMPass(workloadManagementEnable, log, disableDmaSwFifo));
    pm.addPass(ELF::createAddELFSymbolTablePass(log));
    pm.addPass(ELF::createSetEntryPointPass(log));
    pm.addPass(ELF::createAddNetworkMetadataPass(log));
    pm.addPass(VPUASM::createAddProfilingSectionPass(log));
    pm.addPass(VPUASM::createAddCompilerHashPass(log));
}

//
// registerConversionPipelines40XX
//

void vpux::arch40xx::registerConversionPipeline() {
    mlir::PassPipelineRegistration<BackendCompilationOptions40XX>(
            "lower-VPUIP-to-ELF", "Performs full lowering from the VPUIP Dialect to the VPUMI40XX and ELF Dialects",
            [](mlir::OpPassManager& pm, const BackendCompilationOptions40XX& options) {
                vpux::arch40xx::buildLowerVPUIP2ELFPipeline(pm, options);
            });
}
