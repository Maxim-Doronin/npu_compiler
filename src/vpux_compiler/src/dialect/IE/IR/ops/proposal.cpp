//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ProposalOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ProposalOpAdaptor proposal(operands, attrs, prop);
    if (mlir::failed(proposal.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(proposal.getClassProbs().getType());

    // out shape must be [batch_size * post_nms_topn, 5]
    const SmallVector<int64_t> outShape{
            inType.getShape().front() * proposal.getProposalAttrs().getPostNmsTopN().getInt(), 5};
    const SmallVector<int64_t> probsShape{inType.getShape().front() *
                                          proposal.getProposalAttrs().getPostNmsTopN().getInt()};
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(probsShape, inType.getElementType());

    return mlir::success();
}
