//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>

namespace vpux::HostExec {
#define GEN_PASS_DECL_OPTIMIZEMEMREFCOPIES
#define GEN_PASS_DEF_OPTIMIZEMEMREFCOPIES
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {

//
// OptimizeMemRefCopiesPass
//

class OptimizeMemRefCopiesPass final : public HostExec::impl::OptimizeMemRefCopiesBase<OptimizeMemRefCopiesPass> {
public:
    explicit OptimizeMemRefCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeMemRefCopiesPass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::OpBuilder builder(func.getContext());

    //
    // pattern to optimize memref.copy operations that copy from a call result to a subview
    //
    func.walk([&](mlir::memref::CopyOp copyOp) {
        mlir::Value src = copyOp.getSource();
        mlir::Value dst = copyOp.getTarget();

        auto callOp = src.getDefiningOp<mlir::func::CallOp>();
        if (!callOp || callOp.getNumResults() != 1) {
            return;
        }

        auto subviewOp = dst.getDefiningOp<mlir::memref::SubViewOp>();
        if (!subviewOp) {
            return;
        }

        if (callOp.getNumOperands() < 1) {
            return;
        }

        mlir::Value oldDest = callOp.getOperand(callOp.getNumOperands() - 1);

        // Check if the old destination was from an alloc (we want to remove it)
        auto allocOp = oldDest.getDefiningOp<mlir::memref::AllocOp>();
        if (!allocOp) {
            return;
        }

        // Replace call operands to use the subview instead of the alloc
        SmallVector<mlir::Value> newOperands(callOp.getOperands().begin(), callOp.getOperands().end());
        builder.setInsertionPointAfter(subviewOp);
        if (src.getType() != dst.getType()) {
            // E169895: If the types are different, we need to cast the result of the subview to match the call operand
            // type
            auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(callOp.getLoc(), src.getType(), dst);
            newOperands.back() = castOp.getResult(0);
        } else {
            newOperands.back() = subviewOp.getResult();
        }

        auto newCall = builder.create<mlir::func::CallOp>(callOp.getLoc(), callOp.getCallee(), callOp.getResultTypes(),
                                                          newOperands);

        callOp.getResult(0).replaceAllUsesWith(newCall.getResult(0));

        copyOp.erase();
        callOp.erase();
        if (allocOp->use_empty()) {
            allocOp.erase();
        }
    });

    //
    // pattern to optimize memref.copy operations that copy from an alloc to a block argument
    //
    func.walk([&](mlir::memref::CopyOp copyOp) {
        mlir::Value src = copyOp.getSource();
        mlir::Value dst = copyOp.getTarget();

        auto allocOp = src.getDefiningOp<mlir::memref::AllocOp>();
        if (!allocOp) {
            return;
        }

        if (!mlir::isa<mlir::BlockArgument>(dst)) {
            return;
        }

        allocOp.getResult().replaceAllUsesWith(dst);

        copyOp.erase();
        if (allocOp->use_empty()) {
            allocOp.erase();
        }
    });
}

}  // namespace

//
// createOptimizeMemRefCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createOptimizeMemRefCopiesPass(Logger log) {
    return std::make_unique<OptimizeMemRefCopiesPass>(log);
}
