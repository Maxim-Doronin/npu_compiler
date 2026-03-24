//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
#define GEN_PASS_DECL_DECOMPOSEAGGREGATEOPS
#define GEN_PASS_DEF_DECOMPOSEAGGREGATEOPS
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

static mlir::Operation* decomposeSoftmax(mlir::linalg::SoftmaxOp op) {
    mlir::OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);
    auto ctx = builder.getContext();
    auto loc = op.getLoc();
    auto ty = op.getInputOperandType();
    auto elTy = ty.getElementType();
    auto input = op.getOperand(0);
    auto output = op.getOutput();
    auto axis = op.getDimension();
    auto rank = ty.getRank();

    // Pre-compute iterator types and affine maps
    SmallVector<mlir::utils::IteratorType> reductionIterators(rank, mlir::utils::IteratorType::parallel);
    reductionIterators[axis] = mlir::utils::IteratorType::reduction;
    SmallVector<mlir::AffineExpr> reductionAffineExprs;
    for (int64_t i = 0; i < rank; i++) {
        if (i != checked_cast<int64_t>(axis)) {
            reductionAffineExprs.push_back(mlir::getAffineDimExpr(i, ctx));
        }
    }
    auto reductionMap = mlir::AffineMap::get(rank, 0, reductionAffineExprs, ctx);

    SmallVector<mlir::utils::IteratorType> parallelIterators(rank, mlir::utils::IteratorType::parallel);
    auto parallelMap = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<mlir::AffineMap> reductionMaps{parallelMap, reductionMap};
    SmallVector<mlir::AffineMap> broadcastMaps{parallelMap, reductionMap, parallelMap};
    SmallVector<mlir::OpFoldResult> dims = mlir::tensor::getMixedSizes(builder, loc, input);
    dims.erase(dims.begin() + axis);

    // REDUCE MAX
    auto maxFillVal = mlir::arith::getIdentityValue(mlir::arith::AtomicRMWKind::maximumf, elTy, builder, loc,
                                                    /*useOnlyFiniteValue=*/true);
    auto maxEmpt = builder.create<mlir::tensor::EmptyOp>(loc, dims, elTy);
    auto maxInit =
            builder.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{maxFillVal}, mlir::ValueRange(maxEmpt)).result();
    auto max = builder.create<mlir::linalg::GenericOp>(
                              loc, maxEmpt.getType(), input, maxInit, reductionMaps, reductionIterators,
                              [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                                  auto result =
                                          b.create<mlir::arith::MaximumFOp>(
                                                   loc, args[0], args[1],
                                                   mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz)
                                                  .getResult();
                                  b.create<mlir::linalg::YieldOp>(loc, result);
                              })
                       ->getResult(0);

    // SUB+EXP
    auto subExp = builder.create<mlir::linalg::GenericOp>(
                                 loc, output.getType(), mlir::ValueRange{input, max}, output, broadcastMaps,
                                 parallelIterators,
                                 [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                                     auto result = b.create<mlir::arith::SubFOp>(loc, args[0], args[1]).getResult();
                                     result = b.create<mlir::math::ExpOp>(loc, result, mlir::arith::FastMathFlags::afn)
                                                      .getResult();
                                     b.create<mlir::linalg::YieldOp>(loc, result);
                                 })
                          ->getResult(0);

    // REDUCE ADD
    mlir::Type accumElTy = elTy;
    auto addEmpt = maxEmpt;
    if (elTy.getIntOrFloatBitWidth() < 32) {
        accumElTy = mlir::Float32Type::get(ctx);
        addEmpt = builder.create<mlir::tensor::EmptyOp>(loc, dims, accumElTy);
    }
    auto zero = mlir::arith::getIdentityValue(mlir::arith::AtomicRMWKind::addf, accumElTy, builder, loc,
                                              /*useOnlyFiniteValue=*/true);
    auto reduceAddInit =
            builder.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{addEmpt}).result();
    auto reduceAdd = builder.create<mlir::linalg::GenericOp>(
                                    loc, addEmpt.getType(), mlir::ValueRange{subExp}, reduceAddInit, reductionMaps,
                                    reductionIterators,
                                    [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                                        mlir::Value result = args[0];
                                        if (accumElTy != elTy) {
                                            result = b.create<mlir::arith::ExtFOp>(loc, accumElTy, args[0]).getResult();
                                        }
                                        result = b.create<mlir::arith::AddFOp>(loc, args[1], result,
                                                                               mlir::arith::FastMathFlags::reassoc)
                                                         .getResult();
                                        b.create<mlir::linalg::YieldOp>(loc, result);
                                    })
                             ->getResult(0);

    // DIV
    return builder.create<mlir::linalg::GenericOp>(
            loc, output.getType(), mlir::ValueRange{subExp, reduceAdd}, output, broadcastMaps, parallelIterators,
            [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                mlir::Value result = args[0];
                if (accumElTy != elTy) {
                    result = b.create<mlir::arith::ExtFOp>(loc, accumElTy, result).getResult();
                }
                result = b.create<mlir::arith::DivFOp>(loc, result, args[1], mlir::arith::FastMathFlags::arcp)
                                 .getResult();
                if (accumElTy != elTy) {
                    result = b.create<mlir::arith::TruncFOp>(loc, elTy, result).getResult();
                }
                b.create<mlir::linalg::YieldOp>(loc, result);
            });
}

class DecomposeAggregateOpsPass final : public impl::DecomposeAggregateOpsBase<DecomposeAggregateOpsPass> {
public:
    explicit DecomposeAggregateOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

class SoftmaxDecompose : public mlir::OpRewritePattern<mlir::linalg::SoftmaxOp> {
public:
    using mlir::OpRewritePattern<mlir::linalg::SoftmaxOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::SoftmaxOp op, mlir::PatternRewriter& rewriter) const final {
        rewriter.replaceOp(op, decomposeSoftmax(op));
        return mlir::success();
    }
};

void DecomposeAggregateOpsPass::safeRunOnModule() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<mlir::linalg::SoftmaxOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SoftmaxDecompose>(&ctx);

    mlir::ModuleOp moduleOp = getOperation();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);

    if (mlir::failed(mlir::applyPartialConversion(swModule, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> ShaveCodeGen::createDecomposeAggregateOpsPass(Logger log) {
    return std::make_unique<DecomposeAggregateOpsPass>(log);
}
