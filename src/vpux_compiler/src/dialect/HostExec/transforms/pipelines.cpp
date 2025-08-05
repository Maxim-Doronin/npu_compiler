//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// registerHostExecPipelines
//

void HostExec::registerHostExecPipelines() {
    mlir::PassPipelineRegistration<>("hostexec-to-llvm",
                                     "Performs full lowering HostExec to LLVM dialect for host compilation",
                                     [](mlir::OpPassManager& pm) {
                                         buildHostExecPipeline(pm);
                                     });
}

void HostExec::buildHostExecPipeline(mlir::OpPassManager& pm, Logger /*log*/) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());

    pm.addPass(HostExec::createSerializeELFToBinaryPass());
    pm.addPass(HostExec::createConvertToLLVMUMDCallsPass());

    pm.addPass(mlir::createCanonicalizerPass(grc));
}
