//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace vpux {
#define GEN_PASS_DECL_ONESHOTBUFFERIZESWKERNELS
#define GEN_PASS_DEF_ONESHOTBUFFERIZESWKERNELS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// OneShotBufferizeSWKernelsPass
//

class OneShotBufferizeSWKernelsPass final :
        public vpux::impl::OneShotBufferizeSWKernelsBase<OneShotBufferizeSWKernelsPass> {
public:
    explicit OneShotBufferizeSWKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void OneShotBufferizeSWKernelsPass::safeRunOnModule() {
    // Run full one shot bufferization on the software kernels module. This module
    // includes DPS ops and therefore needs the one shot analysis to resolve
    // conflicts.
    auto options = vpux::getOneShotBufferizationOptions();
    options.testAnalysisOnly = false;

    mlir::ModuleOp moduleOp = getOperation();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    mlir::bufferization::BufferizationState state;
    if (mlir::failed(mlir::bufferization::runOneShotBufferize(swModule, options, state, /*statistics=*/nullptr))) {
        signalPassFailure();
        return;
    }

    return;
}

}  // namespace

//
// createOneShotBufferizeSWKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createOneShotBufferizeSWKernelsPass(Logger log) {
    return std::make_unique<OneShotBufferizeSWKernelsPass>(log);
}
