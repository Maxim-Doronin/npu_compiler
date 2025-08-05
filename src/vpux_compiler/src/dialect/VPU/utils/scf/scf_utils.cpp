//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/core/attributes/dim.hpp"

using namespace vpux::VPU;

mlir::OpFoldResult vpux::VPU::getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim) {
    const auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());
    if (!outputType.hasStaticShape() && !operation->getOperand(0).hasOneUse()) {
        auto dimUser = llvm::find_if(operation->getOperand(0).getUsers(), [](auto* user) {
            return mlir::isa<mlir::tensor::DimOp>(user);
        });

        if (dimUser != operation->getOperand(0).getUsers().end()) {
            return dimUser->getResult(0);
        }
    }

    mlir::ReifiedRankedShapedTypeDims resultShape;
    if (mlir::failed(reifyResultShapes(builder, operation, resultShape))) {
        return builder.getIndexAttr(outputType.getDimSize(dim));
    }

    return resultShape[0][dim];
}

SCFTileInfo vpux::VPU::getWeightsTableSCFTile(mlir::Type origWeightsTableType, mlir::OpBuilder& builder,
                                              const SCFTileInfo& outputTile) {
    auto origWeightsTableShape = mlir::cast<mlir::ShapedType>(origWeightsTableType).getShape();

    SCFTileInfo weightsTableTile(origWeightsTableShape, builder);
    weightsTableTile.offsets[0] = outputTile.offsets[Dims4D::Act::C.ind()];
    weightsTableTile.shape[0] = outputTile.shape[Dims4D::Act::C.ind()];
    return weightsTableTile;
}

mlir::Range vpux::VPU::solutionForOutputRange(mlir::Location loc, mlir::OpBuilder& builder,
                                              const SCFTileInfo& outputTile, Dim dim, const int64_t kernel,
                                              const int64_t stride, const std::pair<int64_t, int64_t>& origPadding) {
    auto zero = builder.getIndexAttr(0);
    auto one = builder.getIndexAttr(1);
    mlir::Range inputRange = {zero, zero, one};
    mlir::Range outputRange = {outputTile.offsets[dim.ind()], outputTile.shape[dim.ind()], one};

    mlir::AffineExpr s0, d0;
    bindDims(builder.getContext(), d0);
    bindSymbols(builder.getContext(), s0);

    const auto hasPadBefore = origPadding.first != 0;

    // input offset is based on output tile offset and operation's parameters
    // current calculation is
    // offset: max((output offset) * stride - padding, 0).
    // size: (output size - 1) * stride + kernel - padding
    // if operation has padding, the median tile size will be corrected later if needed
    if (!hasPadBefore && stride == 1) {
        inputRange.offset = outputRange.offset;
    } else {
        auto offsetMap = mlir::AffineMap::get(1, 1, {d0 * stride - origPadding.first, s0}, builder.getContext());
        inputRange.offset = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "inputOffset"), offsetMap,
                                                                      {outputRange.offset, zero});
    }

    auto sizeMap = mlir::AffineMap::get(1, 0, {(d0 - 1) * stride + kernel - origPadding.first}, builder.getContext());
    inputRange.size = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "inputSize"), sizeMap,
                                                                  {outputRange.size});

    return inputRange;
}
