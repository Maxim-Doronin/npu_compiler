//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/utils/pass_disabling_execution_context.hpp"

#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <cstdlib>

using namespace vpux;

template <int N>
class CounterPass : public mlir::PassWrapper<CounterPass<N>, mlir::OperationPass<mlir::ModuleOp>> {
public:
    CounterPass(int& counter)
            : _counter(counter),
              _passId("counter-pass" + std::to_string(N)),
              _passName("CounterPass" + std::to_string(N)) {
    }

    llvm::StringRef getName() const override {
        return _passName;
    }

    llvm::StringRef getArgument() const override {
        return _passId;
    }

    void runOnOperation() override {
        _counter += N;
    }

private:
    int& _counter;
    std::string _passId, _passName;
};

static void runTest(mlir::MLIRContext& ctx, mlir::PassManager& pm) {
    mlir::OwningOpRef<mlir::ModuleOp> mod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    ASSERT_TRUE(mlir::succeeded(pm.run(*mod)));
}

using PassDisablingTest = MLIR_UnitBase;

TEST_F(PassDisablingTest, PassIncrementsWhenEnabled) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(PassDisablingExecutionContext(""));

    int counter = 0;

    pm.addPass(std::make_unique<CounterPass<1>>(counter));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 1);
}

TEST_F(PassDisablingTest, PassDoesNotIncrementWhenDisabled) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(PassDisablingExecutionContext("CounterPass1"));

    int counter = 0;

    pm.addPass(std::make_unique<CounterPass<1>>(counter));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 0);
}

TEST_F(PassDisablingTest, MultiplePassesIncrementWhenEnabled) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(PassDisablingExecutionContext(""));

    int counter = 0;
    int counter2 = 0;

    pm.addPass(std::make_unique<CounterPass<1>>(counter));
    pm.addPass(std::make_unique<CounterPass<1>>(counter2));
    pm.addPass(std::make_unique<CounterPass<1>>(counter));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 2);
    EXPECT_EQ(counter2, 1);
}

TEST_F(PassDisablingTest, MultipleDoNotIncrementWhenDisabled) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(PassDisablingExecutionContext("counter-pass1"));

    int counter = 0;
    int counter2 = 0;

    pm.addPass(std::make_unique<CounterPass<1>>(counter));
    pm.addPass(std::make_unique<CounterPass<1>>(counter2));
    pm.addPass(std::make_unique<CounterPass<1>>(counter));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 0);
    EXPECT_EQ(counter2, 0);
}

TEST_F(PassDisablingTest, RegexAlternationMatchesTwoPasses) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(PassDisablingExecutionContext("counter-pass1|counter-pass2"));

    int counter = 0;
    int counter2 = 0;
    int counter3 = 0;

    pm.addPass(std::make_unique<CounterPass<1>>(counter));
    pm.addPass(std::make_unique<CounterPass<2>>(counter2));
    pm.addPass(std::make_unique<CounterPass<3>>(counter3));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 0);
    EXPECT_EQ(counter2, 0);
    EXPECT_EQ(counter3, 3);
}

TEST_F(PassDisablingTest, RegexDotMatchesAll) {
    mlir::MLIRContext ctx(registry);
    mlir::PassManager pm(&ctx);
    ctx.registerActionHandler(PassDisablingExecutionContext("counter-pass."));

    int counter = 0;
    int counter2 = 0;
    int counter3 = 0;

    pm.addPass(std::make_unique<CounterPass<1>>(counter));
    pm.addPass(std::make_unique<CounterPass<2>>(counter2));
    pm.addPass(std::make_unique<CounterPass<3>>(counter3));
    runTest(ctx, pm);

    EXPECT_EQ(counter, 0);
    EXPECT_EQ(counter2, 0);
    EXPECT_EQ(counter3, 0);
}
