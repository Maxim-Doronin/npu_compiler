//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/utils/pass_disabling_execution_context.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_REGISTERPASSDISABLINGEXECUTIONCONTEXT
#define GEN_PASS_DEF_REGISTERPASSDISABLINGEXECUTIONCONTEXT
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// RegisterPassDisablingExecutionContextPass
//

class RegisterPassDisablingExecutionContextPass final :
        public VPU::impl::RegisterPassDisablingExecutionContextBase<RegisterPassDisablingExecutionContextPass> {
public:
    RegisterPassDisablingExecutionContextPass() = default;
    RegisterPassDisablingExecutionContextPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log);

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnModule() final;
};

RegisterPassDisablingExecutionContextPass::RegisterPassDisablingExecutionContextPass(
        const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    Base::initLogger(log, Base::getArgumentName());
    Base::copyOptionValuesFrom(initCompilerOptions);
}

mlir::LogicalResult RegisterPassDisablingExecutionContextPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    return Base::initializeOptions(options, errorHandler);
}

void RegisterPassDisablingExecutionContextPass::safeRunOnModule() {
    if (disabledPassesOpt.hasValue()) {
        auto* ctx = &getContext();
        const auto disabledPasses = disabledPassesOpt.getValue();
        ctx->registerActionHandler(PassDisablingExecutionContext(disabledPasses));
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createRegisterPassDisablingExecutionContextPass(
        const InitCompilerOptions& initCompilerOptions, Logger log) {
    return std::make_unique<RegisterPassDisablingExecutionContextPass>(initCompilerOptions, log);
}
