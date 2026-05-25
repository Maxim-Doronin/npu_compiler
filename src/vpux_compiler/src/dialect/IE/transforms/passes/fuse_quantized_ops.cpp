//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Dialect/Quant/IR/Quant.h>
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEQUANTIZEDOPS
#define GEN_PASS_DEF_FUSEQUANTIZEDOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

namespace vpux {

//
// FuseQuantizedOpsPass
//

class FuseQuantizedOpsPass final : public IE::impl::FuseQuantizedOpsBase<FuseQuantizedOpsPass> {
public:
    explicit FuseQuantizedOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
};

void FuseQuantizedOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto moduleOp = getModuleOp(func);

    mlir::RewritePatternSet patterns(&ctx);

    // register platform specific rewriters using the platform specific strategy
    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getFuseQuantizedOpsStrategy(config::hasEnableSEPtrsOperations(moduleOp));
    strategy->addPatterns(patterns, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace vpux

//
// createFuseQuantizedOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseQuantizedOpsPass(Logger log) {
    return std::make_unique<FuseQuantizedOpsPass>(log);
}
