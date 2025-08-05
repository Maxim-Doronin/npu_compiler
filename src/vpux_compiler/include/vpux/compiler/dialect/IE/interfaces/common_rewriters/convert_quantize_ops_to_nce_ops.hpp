

//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {

//
// QuantizeDequantizeToAvgPool
//

template <class ConcreteOp>
class QuantizeDequantizeToAvgPool final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    QuantizeDequantizeToAvgPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult QuantizeDequantizeToAvgPool<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    auto newPooling = IE::createIdentityAvgPool(originOp.getInput(), originOp.getType(), rewriter, originOp->getLoc());
    rewriter.replaceOp(originOp, newPooling->getResult(0));
    return mlir::success();
}

//
// QuantizeToAddRewriter
//

class QuantizeToAddRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeToAddRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// DequantizeToAddRewriter
//

class DequantizeToAddRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToAddRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// DequantizeToDwRewriter
//

class DequantizeToDwRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToDwRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp DequantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// QuantizeToDwRewriter
//

class QuantizeToDwRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeToDwRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// DequantizeToConvRewriter
//

class DequantizeToConvRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace vpux::IE
