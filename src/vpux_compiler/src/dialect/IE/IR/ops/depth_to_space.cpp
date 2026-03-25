//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Support/LLVM.h>

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     int64_t block_size, IE::DepthToSpaceMode mode) {
    build(builder, state, input, block_size, mode, nullptr);
}

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::IntegerAttr block_size, IE::DepthToSpaceModeAttr mode) {
    build(builder, state, input, block_size, mode, nullptr);
}

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                     mlir::Value input, mlir::IntegerAttr block_size, IE::DepthToSpaceModeAttr mode) {
    build(builder, state, outType, input, block_size, mode, nullptr);
}

mlir::LogicalResult vpux::IE::DepthToSpaceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DepthToSpaceOpAdaptor depthToSpace(operands, attrs, prop);
    if (mlir::failed(depthToSpace.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(depthToSpace.getInput());
    const auto inType = mlir::cast<mlir::ShapedType>(depthToSpace.getInput().getType()).getElementType();
    const auto block_size = depthToSpace.getBlockSize();
    auto paddedChannels = depthToSpace.getPaddedChannels();

    if (!(inType.isF16() || inType.isF32() || inType.isUnsignedInteger(8) ||
          mlir::isa<mlir::quant::QuantizedType>(inType))) {
        return errorAt(loc, "DepthToSpace only support FP16, FP32, U8, quant data type");
    }

    if (inShape.size() < 3) {
        return errorAt(loc, "Invalid input tensor shape, dimension must be greater than 2.");
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    if (inShape[Dims4D::Act::C] % (block_size * block_size) != 0) {
        return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", block_size,
                       inShape[Dims4D::Act::C]);
    }

    if (inShape[Dims4D::Act::C] == mlir::ShapedType::kDynamic) {
        return errorAt(loc, "Input channels dimension is dynamic, cannot infer output shape");
    }

    // TODO E#194482 Add tests for the possibility of relaxing the legacy dynamic batch constraint

    int64_t paddedIC = 0;
    int64_t paddedOC = 0;

    auto blockSizeSquare = block_size * block_size;
    if (paddedChannels.has_value()) {
        paddedIC = paddedChannels.value().getInput() ? paddedChannels.value().getInput().getInt() : 0;
        paddedOC = paddedChannels.value().getOutput() ? paddedChannels.value().getOutput().getInt() : 0;

        auto unpaddedChannels = inShape[Dims4D::Act::C] - paddedIC;
        if (unpaddedChannels % blockSizeSquare != 0) {
            return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", block_size,
                           unpaddedChannels);
        }
    }

    int64_t C_out = checked_cast<int64_t>((inShape[Dims4D::Act::C] - paddedIC) / blockSizeSquare + paddedOC);

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(depthToSpace.getInput().getType());

    auto [outStaticShape, outBounds, outDimMask] = callOnShapeOf(inputType, [&](const auto& inShape) {
        auto outShape = copyShape(inShape);
        outShape[Dims4D::Act::C] = C_out;
        outShape[Dims4D::Act::H] *= block_size;
        outShape[Dims4D::Act::W] *= block_size;
        return splitShapeAndRepresentation(outShape);
    });

    SmallVector<int64_t> outShape(outStaticShape.begin(), outStaticShape.end());
    const auto outDesc =
            vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), outBounds, outDimMask);
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::DepthToSpaceOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());
    // when block_size == 1, fold to input itself
    if (getBlockSize() == 1) {
        return getInput();
    }

    return nullptr;
}

mlir::LogicalResult IE::DepthToSpaceOp::verify() {
    if (getBlockSize() <= 0) {
        return errorAt(*this, "Block size should be a positive integer, while it is {0}", getBlockSize());
    }
    return mlir::success();
}

mlir::LogicalResult IE::DepthToSpaceOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                          mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();
    // Parse attributes
    auto blockSize = getBlockSize();

    const auto inputShapedType = mlir::cast<mlir::ShapedType>(getInput().getType());
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());

    VPUX_THROW_WHEN(inputShapedType.getRank() != 4 || outputShapedType.getRank() != 4,
                    "reify D2S: Unsupported input or output rank: {0} , {1}", inputShapedType.getRank(),
                    outputShapedType.getRank());

    auto makeIndex = [&](int64_t value) {
        return builder.createOrFold<mlir::arith::ConstantIndexOp>(loc, value);
    };

    auto getInputDimVal = [&](int64_t idx, mlir::Location dimLoc) {
        auto inputDim = reifyDim(builder, getInput(), idx, dimLoc);
        auto inputDimVal = mlir::dyn_cast<mlir::Value>(inputDim);
        VPUX_THROW_WHEN(inputDimVal == nullptr, "Failed to reify input dimension {0} for input {1} at location {2}",
                        idx, getInput(), loc);

        return inputDimVal;
    };

    // Use generator functions based on index for each output dimension
    auto computeShapeForDim = [&](int64_t idx) -> mlir::OpFoldResult {
        auto dimLoc = appendLoc(loc, "dim_{0}", idx);

        if (idx == Dims4D::Act::N.ind()) {
            return reifyDim(builder, getInput(), idx, dimLoc);
        } else if (idx == Dims4D::Act::C.ind()) {
            // outC = inC / (blockSize * blockSize)
            auto inputDimVal = getInputDimVal(idx, dimLoc);
            return builder.createOrFold<mlir::arith::DivSIOp>(dimLoc, inputDimVal, makeIndex(blockSize * blockSize));
        } else if (idx == Dims4D::Act::H.ind() || idx == Dims4D::Act::W.ind()) {
            // outHW = inHW * blockSize
            auto inputDimVal = getInputDimVal(idx, dimLoc);

            return builder.createOrFold<mlir::arith::MulIOp>(dimLoc, inputDimVal, makeIndex(blockSize));
        } else {
            VPUX_THROW("Unexpected dimension index {0}", idx);
        }
    };

    SmallVector<mlir::OpFoldResult> outShape;
    for (const auto dim : llvm::seq<int64_t>(0, outputShapedType.getRank())) {
        if (outputShapedType.isDynamicDim(dim)) {
            outShape.push_back(mlir::getValueOrCreateConstantIndexOp(builder, loc, computeShapeForDim(dim)));
        } else {
            outShape.push_back(builder.getIndexAttr(outputShapedType.getDimSize(dim)));
        }
    }

    reifiedReturnShapes.emplace_back(std::move(outShape));
    return mlir::success();
}
