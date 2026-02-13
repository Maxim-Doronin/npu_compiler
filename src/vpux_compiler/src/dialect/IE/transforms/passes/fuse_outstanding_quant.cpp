//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEOUTSTANDINGQUANT
#define GEN_PASS_DEF_FUSEOUTSTANDINGQUANT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

namespace vpux {

//
// FuseOutstandingQuantPass
//

class FuseOutstandingQuantPass final : public IE::impl::FuseOutstandingQuantBase<FuseOutstandingQuantPass> {
public:
    explicit FuseOutstandingQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult FuseOutstandingQuantPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

void FuseOutstandingQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // register platform specific rewriters using the platform specific strategy
    auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getFuseOutstandingQuantStrategy();
    strategy->addPatterns(patterns, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace vpux

//
// createFuseOutstandingQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseOutstandingQuantPass(Logger log) {
    return std::make_unique<FuseOutstandingQuantPass>(log);
}
