//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Dialect/Quant/Transforms/Passes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_STRIPFUNCQUANTTYPESCODEGEN
#define GEN_PASS_DEF_STRIPFUNCQUANTTYPESCODEGEN
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// StripFuncQuantTypesCodeGenPass
//

class StripFuncQuantTypesCodeGenPass final :
        public ShaveCodeGen::impl::StripFuncQuantTypesCodeGenBase<StripFuncQuantTypesCodeGenPass> {
public:
    explicit StripFuncQuantTypesCodeGenPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void StripFuncQuantTypesCodeGenPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    mlir::PassManager pm(&ctx);

    pm.addPass(mlir::quant::createStripFuncQuantTypes());

    if (mlir::failed(pm.run(swModule))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createStripFuncQuantTypesCodeGenPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createStripFuncQuantTypesCodeGenPass(Logger log) {
    return std::make_unique<StripFuncQuantTypesCodeGenPass>(log);
}
