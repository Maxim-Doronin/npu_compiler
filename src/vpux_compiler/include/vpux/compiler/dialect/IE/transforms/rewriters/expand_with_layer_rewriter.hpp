//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

//
// ExpandWithLayer
//

class ExpandWithLayer final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    ExpandWithLayer(mlir::MLIRContext* ctx,
                    const std::function<bool(IE::ExpandOp, mlir::Operation*)>& isBeneficalToSwap, Logger log,
                    mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx, benefit), _isBeneficalToSwap(isBeneficalToSwap), _log(log) {
        setDebugName("ExpandWithLayer");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origExpandOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::function<bool(IE::ExpandOp, mlir::Operation* layerOp)> _isBeneficalToSwap;
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
