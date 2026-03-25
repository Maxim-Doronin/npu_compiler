//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>

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
