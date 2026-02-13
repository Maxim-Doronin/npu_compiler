//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/reshape_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::GenericReshapeOp::verify() {
    const auto op = getOperation();
    auto distributedInType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(getInput().getType());
    auto distributedOutType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(getOutput().getType());
    if (distributedInType && distributedOutType) {
        if (!isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(distributedInType,
                                                                                             distributedOutType)) {
            return errorAt(op, "Reshape has incompatible output shape as clustering: in type = {0}, out type = {1}",
                           distributedInType, distributedOutType);
        }
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "Reshape input and output must have the same number of elements");
    }

    if (!isInAndOutStridesCompatible(inType, outType)) {
        return errorAt(op, "Incompatible strides between input {0} and output {1}", inType, outType);
    }

    return mlir::success();
}

mlir::Value VPUIP::GenericReshapeOp::getViewSource() {
    return getInput();
}

mlir::OpFoldResult VPUIP::GenericReshapeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    if (const auto cst = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return static_cast<Const::ContentAttr>(cst).transform().reshape(getShape(getOutput())).get();
    }

    return nullptr;
}

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<VPUIP::GenericReshapeOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::GenericReshapeOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(VPUIP::GenericReshapeOp origOp,
                                                  mlir::PatternRewriter& rewriter) const {
    auto producerReshapeOp = origOp.getInput().getDefiningOp<VPUIP::GenericReshapeOp>();
    if (producerReshapeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::GenericReshapeOp>(origOp, origOp.getOutput().getType(),
                                                         producerReshapeOp.getInput());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::GenericReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseReshapes>(ctx);
}
