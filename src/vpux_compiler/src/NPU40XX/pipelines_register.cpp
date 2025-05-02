//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/pipelines_register.hpp"
#include "vpux/compiler/NPU40XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/pipeline_strategies.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry40XX::registerPipelines
//

void PipelineRegistry40XX::registerPipelines() {
    mlir::PassPipelineRegistration<DefaultHWOptions40XX>(
            "ShaveCodeGen", "Compile both from IE to VPUIP and from IERT to LLVM for VPU40XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions40XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{VPU::ArchKind::NPU40XX, VPU::CompilationMode::ShaveCodeGen,
                                                             options};
                auto createPipelineStartegy = [&](VPU::CompilationMode) {
                    return createDialectPipelineStrategy40XX<DefaultHWOptions40XX>(&initCompilerOptions, &options);
                };
                ShaveCodeGenStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<ReferenceSWOptions40XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for VPU40XX",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions40XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{VPU::ArchKind::NPU40XX, VPU::CompilationMode::ReferenceSW,
                                                             options};
                auto createPipelineStartegy = [&](VPU::CompilationMode) {
                    return createDialectPipelineStrategy40XX<ReferenceSWOptions40XX>(&initCompilerOptions, &options);
                };
                ReferenceSWStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions40XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for VPU40XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions40XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{VPU::ArchKind::NPU40XX, VPU::CompilationMode::DefaultHW,
                                                             options};
                auto createPipelineStartegy = [&](VPU::CompilationMode) {
                    return createDialectPipelineStrategy40XX<DefaultHWOptions40XX>(&initCompilerOptions, &options);
                };
                DefaultHwStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    vpux::IE::arch40xx::registerIEPipelines();
    vpux::VPU::arch40xx::registerVPUPipelines();
    vpux::VPUIP::arch40xx::registerVPUIPPipelines();
    vpux::arch40xx::registerConversionPipeline();
}
