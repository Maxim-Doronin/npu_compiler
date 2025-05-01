//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpuasm2npureg40xx {

class ManagedMappedInferenceRewriter final : public mlir::OpRewritePattern<VPUASM::ManagedMappedInferenceOp> {
public:
    ManagedMappedInferenceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ManagedMappedInferenceOp>(ctx), _log(log) {
        setDebugName("ManagedMappedInference_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ManagedMappedInferenceOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
