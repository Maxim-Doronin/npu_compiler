//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// ShaveCodeGen
//

void vpux::ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createConvertEltwiseLayers2MathPass(log));
    pm.addPass(mlir::createConvertElementwiseToLinalgPass());
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-sw-layers-to-linalg",
                                     "Performs full lowering of compatible SW Layers from IE to Linalg",
                                     [](mlir::OpPassManager& pm) {
                                         ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(pm);
                                     });
}
