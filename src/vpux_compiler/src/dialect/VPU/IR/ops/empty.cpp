//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"

using namespace vpux;

namespace {

class FuseSlice final : public mlir::OpRewritePattern<VPU::EmptyOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::EmptyOp op, mlir::PatternRewriter& rewriter) const final {
        SmallVector<mlir::Operation*> sliceUsers;
        for (auto userOp : llvm::make_early_inc_range(op->getUsers())) {
            if (mlir::isa<VPU::SliceOp>(userOp)) {
                sliceUsers.push_back(userOp);
            }
        }
        if (sliceUsers.empty()) {
            return mlir::failure();
        }
        for (auto sliceUser : sliceUsers) {
            rewriter.replaceOpWithNewOp<VPU::EmptyOp>(sliceUser, sliceUser->getResult(0).getType());
        }
        return mlir::success();
    }
};

}  // namespace

void VPU::EmptyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseSlice>(ctx);
}
