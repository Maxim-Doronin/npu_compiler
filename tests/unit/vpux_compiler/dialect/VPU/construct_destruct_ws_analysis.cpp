//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;
using MLIR_VPU_ConstructDestructWsAnalysis = MLIR_UnitBase;

namespace {
constexpr StringLiteral inputIR = R"(
{-#
  dialect_resources: {
    builtin: {
            ov_0: "0x10000000ABCDABCDABCDABCE",
            ov_1: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

module @Test {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x2x2xf16>
    }

    func.func @main() -> tensor<1x1x2x2xf16> {
        %cst = const.Declare tensor<1x1x2x2xf16> = dense_resource<ov_0> : tensor<1x1x2x2xf16>, [#const.Rescale<42.0>]
        %cst2 = const.Declare tensor<1x1x2x2xf16> = dense_resource<ov_1> : tensor<1x1x2x2xf16>, [#const.Rescale<42.0>]
        return %cst : tensor<1x1x2x2xf16>
    }
})";
}  // namespace

namespace {
// Checks that the analysis object is cached or not cached.
class CheckCachePass : public mlir::PassWrapper<CheckCachePass, vpux::ModulePass> {
    bool _shouldBeCached{false};

public:
    CheckCachePass(bool shouldBeCached): _shouldBeCached(shouldBeCached) {
    }

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckCachePass);

    ::llvm::StringRef getName() const override {
        return "CheckCachePass";
    }

    void safeRunOnModule() final {
        auto maybeAnalysis = getCachedAnalysis<VPU::WeightsSeparationInfo>();
        const bool validResult = (_shouldBeCached == maybeAnalysis.has_value());
        if (!validResult) {
            llvm::errs() << "Analysis object must " << (_shouldBeCached ? "" : "NOT ") << "be preserved, but it is "
                         << (maybeAnalysis.has_value() ? "" : "NOT ") << "preserved\n";
            llvm::errs().flush();
            signalPassFailure();
        }
    }
};
}  // namespace

TEST(MLIR_VPU_ConstructDestructWsAnalysis, EnsureRaii) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    auto m = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(m.get() != nullptr);

    // Note: for the sake of this test, any options would suffice
    VPU::WeightsSeparationInfo::setOptions(m.get(), VPU::WeightsSeparationInfo::Options{});
    VPUX_SCOPE_EXIT {
        VPU::WeightsSeparationInfo::removeOptions(m.get());
    };

    mlir::PassManager pm(m.get()->getName());
    pm.addPass(std::make_unique<CheckCachePass>(/*should be cached*/ false));  // no caching before
    pm.addPass(VPU::createConstructWsAnalysisPass());
    pm.addPass(std::make_unique<CheckCachePass>(/*should be cached*/ true));
    pm.addPass(VPU::createDestructWsAnalysisPass());
    pm.addPass(std::make_unique<CheckCachePass>(/*should be cached*/ false));

    ASSERT_TRUE(mlir::succeeded(pm.run(m.get())));
}
