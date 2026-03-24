//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_OUTLINECODEGENCAPSULES
#define GEN_PASS_DEF_OUTLINECODEGENCAPSULES
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {
mlir::func::FuncOp outlineSwLayer(mlir::MLIRContext* ctx, mlir::ModuleOp module, IE::CodeGenCapsuleOp capsuleOp,
                                  size_t counter) {
    auto capsuleBlock = capsuleOp.getBody();
    auto dpsInputs = vpux::to_small_vector(capsuleBlock->getArgumentTypes());

    auto capsuleTerminator = mlir::cast<IE::CGCYieldOp>(capsuleBlock->getTerminator());
    auto outputTypes = vpux::to_small_vector(capsuleTerminator->getOperandTypes());

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());

    builder.setInsertionPointToEnd(capsuleBlock);
    auto loc = capsuleTerminator->getLoc();

    builder.create<mlir::func::ReturnOp>(loc, capsuleTerminator->getOperands());
    capsuleTerminator->erase();

    // Actually create the function
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = mlir::FunctionType::get(ctx, dpsInputs, {outputTypes});
    auto funcName = printToString("generated_{0}", counter);
    auto funcOp = builder.create<mlir::func::FuncOp>(loc, funcName, funcType);
    auto funcOpBody = funcOp.addEntryBlock();
    capsuleBlock->dropAllUses();
    capsuleBlock->moveBefore(funcOpBody);
    funcOpBody->erase();

    return funcOp;
}

static VPU::GenericSwLayerOp createSwLayerOp(mlir::OpBuilder& builder, IE::CodeGenCapsuleOp op,
                                             mlir::SymbolRefAttr fullSymRef) {
    return builder.create<VPU::GenericSwLayerOp>(op->getLoc(), op->getResultTypes(), fullSymRef, op->getOperands());
}

//
// OutlineCodeGenCapsulesPass
//

class OutlineCodeGenCapsulesPass final :
        public ShaveCodeGen::impl::OutlineCodeGenCapsulesBase<OutlineCodeGenCapsulesPass> {
public:
    explicit OutlineCodeGenCapsulesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnModule() final;
};

struct OutlineCodeGenCapsule : mlir::OpRewritePattern<IE::CodeGenCapsuleOp> {
    using OpRewritePattern::OpRewritePattern;

    explicit OutlineCodeGenCapsule(mlir::MLIRContext* ctx, const mlir::ModuleOp swModule, size_t& counter,
                                   mlir::StringAttr swModuleRoot)
            : OpRewritePattern<IE::CodeGenCapsuleOp>(ctx),
              _swModule(swModule),
              _counter(counter),
              _swModuleRoot(swModuleRoot) {
    }

    mlir::LogicalResult matchAndRewrite(IE::CodeGenCapsuleOp op, mlir::PatternRewriter& rewriter) const override {
        auto& ctx = *rewriter.getContext();

        auto outlinedFunc = outlineSwLayer(&ctx, _swModule, op, _counter);
        mlir::OpBuilder builder(&ctx);
        builder.setInsertionPointAfter(op);
        auto fullSymRef = mlir::SymbolRefAttr::get(
                _swModuleRoot, llvm::ArrayRef<mlir::FlatSymbolRefAttr>(mlir::FlatSymbolRefAttr::get(outlinedFunc)));

        auto genericSwLayerOp = createSwLayerOp(builder, op, fullSymRef);
        rewriter.replaceOp(op, genericSwLayerOp->getResults());

        _counter++;
        return mlir::success();
    }

private:
    mlir::ModuleOp _swModule;
    size_t& _counter;
    mlir::StringAttr _swModuleRoot;
};

void OutlineCodeGenCapsulesPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto moduleOp = getOperation();
    auto func = net::getMainFunc(moduleOp);

    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    size_t counter = 0;

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<OutlineCodeGenCapsule>(&ctx, swModule, counter, swModule.getSymNameAttr());
    if (failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
    }
}

}  // namespace

//
// createOutlineCodeGenCapsulesPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createOutlineCodeGenCapsulesPass(Logger log) {
    return std::make_unique<OutlineCodeGenCapsulesPass>(log);
}
