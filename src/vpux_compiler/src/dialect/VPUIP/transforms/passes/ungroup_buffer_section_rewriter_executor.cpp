//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/factories/ungroup_buffer_section_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>

namespace vpux {
#define GEN_PASS_DECL_UNGROUPBUFFERSECTIONREWRITEREXECUTOR
#define GEN_PASS_DEF_UNGROUPBUFFERSECTIONREWRITEREXECUTOR
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// UngroupBufferSectionRewriterExecutorPass
//

class UngroupBufferSectionRewriterExecutorPass final :
        public impl::UngroupBufferSectionRewriterExecutorBase<UngroupBufferSectionRewriterExecutorPass>,
        public RewriterExecutorInterface {
public:
    using Base = impl::UngroupBufferSectionRewriterExecutorBase<UngroupBufferSectionRewriterExecutorPass>;

    explicit UngroupBufferSectionRewriterExecutorPass(bool enableReorderSubViewOp, Logger log)
            : _enableReorderSubViewOp(enableReorderSubViewOp) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnFunc() final;
    bool _enableReorderSubViewOp = false;
};

mlir::LogicalResult UngroupBufferSectionRewriterExecutorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (rewriterName.hasValue()) {
        setRewriterName(rewriterName.getValue());
    }

    if (enableReorderSubViewOp.hasValue()) {
        _enableReorderSubViewOp = enableReorderSubViewOp.getValue();
    }

    return mlir::success();
}

void UngroupBufferSectionRewriterExecutorPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    auto strategy = VPUIP::createUngroupBufferSectionStrategy(_enableReorderSubViewOp);
    auto _customRegistry = vpux::RegistryManager::createCustomRegistry();
    strategy->registerRewriters(*_customRegistry, _log);

    if (mlir::failed(this->executeRewriters(&ctx, _log, func, _customRegistry.get()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupBufferSectionRewriterExecutorPass(bool enableReorderSubViewOp,
                                                                                        Logger log) {
    return std::make_unique<UngroupBufferSectionRewriterExecutorPass>(enableReorderSubViewOp, log);
}

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupBufferSectionRewriterExecutorPass(Logger log) {
    return std::make_unique<UngroupBufferSectionRewriterExecutorPass>(false, log);
}
