//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Interfaces/CallInterfaces.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_WRAPFUNCCALLSINTOASYNCREGIONS
#define GEN_PASS_DEF_WRAPFUNCCALLSINTOASYNCREGIONS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

void wrapIntoAsyncRegion(mlir::Operation* op, Logger log) {
    if (op->getParentOfType<mlir::async::ExecuteOp>() != nullptr) {
        log.trace("[SKIP] The Operation already wrapped into asynchronous region");
        return;
    }

    mlir::Value group = nullptr;
    if (auto forOp = op->getParentOfType<mlir::scf::ForOp>()) {
        mlir::OpBuilder builder(forOp);

        builder.setInsertionPoint(forOp);
        auto step = forOp.getStep();
        auto lb = forOp.getLowerBound();
        auto ub = forOp.getUpperBound();
        auto numberOfIterations = builder.create<mlir::arith::SubIOp>(forOp.getLoc(), ub, lb);
        auto numberOfIterationsDivStep = builder.create<mlir::arith::DivSIOp>(forOp.getLoc(), numberOfIterations, step);
        group = builder.create<mlir::async::CreateGroupOp>(forOp.getLoc(), numberOfIterationsDivStep);

        builder.setInsertionPointAfter(forOp);
        builder.create<mlir::async::AwaitAllOp>(forOp.getLoc(), group);
    }

    log.trace("Create 'async.execute' Operation");

    const auto bodyBuilder = [op](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange) {
        auto* newOp = builder.clone(*op);
        builder.create<mlir::async::YieldOp>(loc, newOp->getResults());
    };

    OpBuilderLogger builderLog(log.nest());
    mlir::OpBuilder builder(op, &builderLog);

    auto execOp = builder.create<mlir::async::ExecuteOp>(op->getLoc(), op->getResultTypes(), std::nullopt, std::nullopt,
                                                         bodyBuilder);
    if (auto forOp = op->getParentOfType<mlir::scf::ForOp>(); forOp != nullptr) {
        builder.create<mlir::async::AddToGroupOp>(op->getLoc(), execOp.getToken(), group);
    }

    log.trace("Create 'async.await' Operations per each original result");

    SmallVector<mlir::Value> newResults;
    newResults.resize(op->getNumResults());
    for (auto i : irange(op->getNumResults())) {
        auto waitOp = builder.create<mlir::async::AwaitOp>(op->getLoc(), execOp.getBodyResults()[i]);
        newResults[i] = waitOp.getResult();
    }

    log.trace("Replace the operation with new 'async.await' results");

    op->replaceAllUsesWith(newResults);
    op->erase();
}

//
// WrapIntoAsyncRegionsPass
//

class WrapFuncCallsIntoAsyncRegionsPass final :
        public VPUIP::impl::WrapFuncCallsIntoAsyncRegionsBase<WrapFuncCallsIntoAsyncRegionsPass> {
public:
    explicit WrapFuncCallsIntoAsyncRegionsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WrapFuncCallsIntoAsyncRegionsPass::safeRunOnFunc() {
    const auto callback = [&](mlir::Operation* op) {
        _log.trace("Process Layer Operation '{0}' at '{1}'", op->getName(), op->getLoc());
        if (mlir::isa<mlir::CallOpInterface>(op)) {
            wrapIntoAsyncRegion(op, _log.nest());
        }
    };

    getOperation().walk(callback);
}

}  // namespace

//
// createWrapIntoAsyncRegionsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createWrapFuncCallsIntoAsyncRegionsPass(Logger log) {
    return std::make_unique<WrapFuncCallsIntoAsyncRegionsPass>(log);
}
