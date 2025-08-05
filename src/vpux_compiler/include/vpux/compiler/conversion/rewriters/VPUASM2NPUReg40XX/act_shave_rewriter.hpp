//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpuasm2npureg40xx {

class ActShaveRtRewriter final : public mlir::OpRewritePattern<VPUASM::ActShaveRtOp> {
public:
    ActShaveRtRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ActShaveRtOp>(ctx), _log(log) {
        setDebugName("ActShaveRt_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ActShaveRtOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
