//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/RegionUtils.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_OUTLINELINALGSWLAYERS
#define GEN_PASS_DEF_OUTLINELINALGSWLAYERS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {
mlir::func::FuncOp outlineSwLayer(mlir::MLIRContext* ctx, mlir::ModuleOp module, mlir::linalg::GenericOp layerOp,
                                  size_t counter) {
    auto dpsInputs = vpux::to_small_vector(layerOp.getInputs().getTypes());
    auto outputTypes = vpux::to_small_vector(layerOp.getOutputs().getTypes());

    // Preserve DPS for the function arguments
    dpsInputs.append(vpux::to_small_vector(outputTypes));
    auto funcType = mlir::FunctionType::get(ctx, dpsInputs, outputTypes);

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());

    auto funcName = printToString("generated_{0}", counter);
    auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcName, funcType);
    auto funcOpBody = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(funcOpBody);

    auto newLayerOp = builder.clone(*layerOp.getOperation());
    newLayerOp->setOperands(funcOpBody->getArguments());

    // Need to return the value produced by the computation ops, otherwise they will get removed at canonicalization
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), newLayerOp->getResults());
    return funcOp;
}

// Replace op with genericSwLayerOp, performing bitcasts for any mismatched output type.
static void replaceOpWithCoercedOutputs(mlir::PatternRewriter& rewriter, mlir::OpBuilder& builder,
                                        mlir::linalg::GenericOp op, VPU::GenericSwLayerOp genericSwLayerOp) {
    SmallVector<mlir::Value> results;

    for (auto result : op.getResults()) {
        auto index = result.getResultNumber();
        auto swLayerResult = genericSwLayerOp.getResult(index);
        if (swLayerResult.getType() != result.getType()) {
            auto cast = builder.create<mlir::tensor::BitcastOp>(op->getLoc(), result.getType(), swLayerResult);
            results.push_back(cast);
            continue;
        }
        results.push_back(swLayerResult);
    }
    rewriter.replaceOp(op, results);
}

// Create a GenericSwLayerOp for the outlined op. Types for the GenericSwLayerOp are chosen
// to allow removal of bitcasts used to convert from signless integers (used by the outlined
// op) to signed/unsigned integers.
static VPU::GenericSwLayerOp createSwLayerOp(mlir::OpBuilder& builder, mlir::linalg::GenericOp op,
                                             mlir::SymbolRefAttr fullSymRef) {
    // Construct the operands for the new GenericSwLayerOp. Peek through input integer bitcasts
    // to deduce the possibly signed/unsigned input tensors.
    auto swKernOperands = llvm::map_to_vector(op.getInputs(), [&](mlir::Value operand) {
        if (operand.getType().isIntOrIndexOrFloat()) {
            // Maintain types for any scalar operands.
            return operand;
        }
        if (mlir::isa<mlir::FloatType>(mlir::cast<vpux::NDTypeInterface>(operand.getType()).getElementType())) {
            // We don't expect any bitcasts on float tensors.
            return operand;
        }
        while (auto cast = mlir::dyn_cast_or_null<mlir::tensor::BitcastOp>(operand.getDefiningOp())) {
            // Look through any bitcasts to find the actual type.
            operand = cast.getOperand();
        }
        return operand;
    });

    // Construct the result types for the GenericSwLayerOp. If the results are integer tensors
    // look for any bitcasts (these should convert to the type of the original op).
    auto swKernResType = llvm::map_to_vector(op.getResults(), [&](mlir::Value operand) {
        // GenericSwLayerOps can only return tensors.
        if (mlir::isa<mlir::FloatType>(mlir::cast<vpux::NDTypeInterface>(operand.getType()).getElementType())) {
            return operand.getType();
        }
        for (auto* user : operand.getUsers()) {
            if (auto cast = mlir::dyn_cast<mlir::tensor::BitcastOp>(user)) {
                mlir::Type retType = cast.getType();
                return retType;
            }
        }
        return operand.getType();
    });

    return builder.create<VPU::GenericSwLayerOp>(op->getLoc(), swKernResType, fullSymRef, swKernOperands);
}

//
// OutlineLinalgSwLayersPass
//

class OutlineLinalgSwLayersPass final :
        public ShaveCodeGen::impl::OutlineLinalgSwLayersBase<OutlineLinalgSwLayersPass> {
public:
    explicit OutlineLinalgSwLayersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnModule() final;
};

struct OutlineLinalgLayer : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    explicit OutlineLinalgLayer(mlir::MLIRContext* ctx, const mlir::ModuleOp swModule, size_t& counter,
                                mlir::StringAttr swModuleRoot)
            : OpRewritePattern<mlir::linalg::GenericOp>(ctx),
              swModule(swModule),
              counter(counter),
              swModuleRoot(swModuleRoot) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op, mlir::PatternRewriter& rewriter) const override {
        auto& ctx = *rewriter.getContext();
        makeRegionIsolatedFromAbove(rewriter, op.getRegion(), [](mlir::Operation* toClone) {
            return mlir::isa<mlir::arith::ConstantOp>(toClone);
        });
        auto outlinedFunc = outlineSwLayer(&ctx, swModule, op, counter);
        mlir::OpBuilder builder(&ctx);
        builder.setInsertionPointAfter(op);
        auto fullSymRef = mlir::SymbolRefAttr::get(
                swModuleRoot, llvm::ArrayRef<mlir::FlatSymbolRefAttr>(mlir::FlatSymbolRefAttr::get(outlinedFunc)));

        auto genericSwLayerOp = createSwLayerOp(builder, op, fullSymRef);
        replaceOpWithCoercedOutputs(rewriter, builder, op, genericSwLayerOp);
        counter++;
        return mlir::success();
    }

private:
    mlir::ModuleOp swModule;
    size_t& counter;
    mlir::StringAttr swModuleRoot;
};

void OutlineLinalgSwLayersPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto moduleOp = getOperation();
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp func;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, func);

    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    size_t counter = 0;

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<OutlineLinalgLayer>(&ctx, swModule, counter, swModule.getSymNameAttr());
    mlir::tensor::BitcastOp::getCanonicalizationPatterns(patterns, &ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
    }
}

}  // namespace

//
// createOutlineLinalgSwLayersPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createOutlineLinalgSwLayersPass(Logger log) {
    return std::make_unique<OutlineLinalgSwLayersPass>(log);
}
