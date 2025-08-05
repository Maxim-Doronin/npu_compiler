//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/convert_to_palletization_lut_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTOPALLETIZATIONLUT
#define GEN_PASS_DEF_CONVERTTOPALLETIZATIONLUT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertToPalletizationLUT
//

class ConvertToPalletizationLUT final : public IE::impl::ConvertToPalletizationLUTBase<ConvertToPalletizationLUT> {
public:
    explicit ConvertToPalletizationLUT(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertToPalletizationLUT::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    mlir::ConversionTarget target(ctx);
    // register platform specific rewriters using the platform specific strategy
    auto strategy = vpux::IE::createConvertToPalletizationLUTStrategy(func);
    strategy->addPatterns(patterns, _log);
    strategy->markOpLegality(target, _log);

    if (mlir::failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToPalletizationLUT
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToPalletizationLUT(Logger log) {
    return std::make_unique<ConvertToPalletizationLUT>(log);
}
