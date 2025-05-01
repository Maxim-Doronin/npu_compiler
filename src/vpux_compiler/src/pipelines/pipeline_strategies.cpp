//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/pipeline_strategies.hpp"
#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/conversion.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// DefaultHwStrategy
//

void DefaultHwStrategy::buildPipeline(mlir::OpPassManager& pm) {
    auto strategy = _createPipelineStrategy(VPU::CompilationMode::DefaultHW);

    strategy->initializePipeline(pm, _log);
    strategy->buildIEPipeline(pm, _log);
    strategy->buildLowerIE2VPUPipeline(pm, _log);
    strategy->buildVPUPipeline(pm, _log);
    strategy->buildLowerVPU2VPUIPPipeline(pm, _log);
    strategy->buildVPUIPPipeline(pm, _log);
}

//
// ShaveCodeGenStrategy
//

void ShaveCodeGenStrategy::buildPipeline(mlir::OpPassManager& pm) {
    _log.trace("Entered buildShaveCodeGenPipeline()");
    auto strategy = _createPipelineStrategy(VPU::CompilationMode::ShaveCodeGen);

    strategy->initializePipeline(pm, _log);
    strategy->buildIEPipeline(pm, _log);

    ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(pm, _log);
    pm.addPass(ShaveCodeGen::createOutlineLinalgSwLayersPass());

    strategy->buildLowerIE2VPUPipeline(pm, _log);
    strategy->buildVPUPipeline(pm, _log);
    strategy->buildLowerVPU2VPUIPPipeline(pm, _log);

    pm.addPass(
            mlir::createConvertLinalgToAffineLoopsPass());  // E#154403 Analyze the pros/cons & replace Affine with SCF
    pm.addPass(ShaveCodeGen::createConvertAffine2LLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(ShaveCodeGen::createAdaptLLVMFuncsForShavePass());

    strategy->buildVPUIPPipeline(pm, _log);

    _log.trace("Entered buildShaveCodeGenPipeline()");
}

//
// ReferenceSWStrategy
//

void ReferenceSWStrategy::buildPipeline(mlir::OpPassManager& pm) {
    auto strategy = _createPipelineStrategy(VPU::CompilationMode::ReferenceSW);

    strategy->initializePipeline(pm, _log);
    strategy->buildReferenceSWPipeline(pm, _log);
}

//
// WSMonolithicStrategy
//

void WSMonolithicStrategy::buildPipeline(mlir::OpPassManager&) {
    VPUX_THROW("Not implemented: E#156402");
}

//
// HostPipelineStrategy
//

void HostPipelineStrategy::buildPipeline(mlir::OpPassManager&) {
    VPUX_THROW("Not implemented: E#157476");
}

//
// createPipelineFactory
//

std::unique_ptr<IFrontendPipelineStrategy> createPipelineFactory(VPU::CompilationMode compilationMode,
                                                                 StrategyFactoryFn createPipelineStrategy, Logger log) {
    switch (compilationMode) {
    case VPU::CompilationMode::DefaultHW:
        return std::make_unique<DefaultHwStrategy>(createPipelineStrategy, log);
    case VPU::CompilationMode::ReferenceSW:
        return std::make_unique<ReferenceSWStrategy>(createPipelineStrategy, log);
    case VPU::CompilationMode::ShaveCodeGen:
        return std::make_unique<ShaveCodeGenStrategy>(createPipelineStrategy, log);
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

}  // namespace vpux
