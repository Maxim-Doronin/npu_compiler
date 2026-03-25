//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>

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
    explicit FuseQuantizedOpsPass(const bool seOpsEnabled, const bool seExperimentalOpsEnabled, Logger log)
            : _seOpsEnabled(seOpsEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;
};

mlir::LogicalResult FuseQuantizedOpsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }

    if (seExperimentalOpsEnabled.hasValue()) {
        _seExperimentalOpsEnabled = seExperimentalOpsEnabled.getValue();
    }

    return mlir::success();
}

void FuseQuantizedOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // register platform specific rewriters using the platform specific strategy
    auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getFuseQuantizedOpsStrategy(_seOpsEnabled, _seExperimentalOpsEnabled);
    strategy->addPatterns(patterns, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace vpux

//
// createFuseQuantizedOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseQuantizedOpsPass(const bool seOpsEnabled,
                                                                 const bool seExperimentalOpsEnabled, Logger log) {
    return std::make_unique<FuseQuantizedOpsPass>(seOpsEnabled, seExperimentalOpsEnabled, log);
}
