//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

namespace vpux {
namespace ShaveCodeGen {

mlir::Type getLinalgElementType(mlir::Type ty, mlir::MLIRContext* ctx) {
    assert(mlir::isa<mlir::RankedTensorType>(ty) && "Ranked tensor type required for getLinalgElementType");
    // Get the math/arith compatible element type for the ty tensor type.
    // math/arith dialects don't accept non-signless integers.
    auto elTy = mlir::cast<mlir::RankedTensorType>(ty).getElementType();

    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elTy)) {
        if (!intTy.isSignless()) {
            return mlir::IntegerType::get(ctx, intTy.getWidth());
        }
        return elTy;
    }

    // We need to convert the quantized type to its storage type because quant types
    // are not part of the math or arith dialects in MLIR.
    if (auto quantTy = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elTy)) {
        auto storageType = quantTy.getStorageType();
        if (auto storageIntTy = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
            // If the storage type is signless, return as it is, otherwise convert to signless
            return storageIntTy.isSignlessInteger() ? storageIntTy
                                                    : mlir::IntegerType::get(ctx, storageIntTy.getWidth());
        } else {
            // Non-integer storage types are not supported
            VPUX_THROW("Non int storage types not supported");
        }
    }

    return elTy;
}

mlir::RankedTensorType getUnpaddedTensorType(mlir::RankedTensorType type, mlir::Location loc,
                                             std::optional<mlir::ArrayAttr> padding) {
    if (!padding) {
        return type;
    }

    auto outShape = SmallVector<int64_t>(type.getShape());
    if (mlir::failed(IE::unpadInputShape(outShape, *padding, loc))) {
        return nullptr;
    }
    auto unpaddedOutTy = mlir::cast<NDTypeInterface>(type).changeShape(ShapeRef(outShape));
    return mlir::cast<mlir::RankedTensorType>(unpaddedOutTy);
}

mlir::RankedTensorType normalizeType(mlir::RankedTensorType type) {
    auto rank = type.getRank();
    auto ndTy = mlir::cast<vpux::NDTypeInterface>(type);
    auto signlessElTy = getLinalgElementType(type, type.getContext());

    auto dstShape = Shape(ndTy.getDimsOrder().toMemoryOrder(ndTy.getShape()).raw());
    auto retTy = ndTy.changeDimsOrder(DimsOrder::fromNumDims(rank)).changeShape(dstShape);
    retTy = retTy.changeElemType(signlessElTy);
    return mlir::cast<mlir::RankedTensorType>(retTy);
}

mlir::Value removePadding(mlir::Value operand, mlir::Value typeConvertedOperand, mlir::PatternRewriter& rewriter,
                          std::optional<mlir::ArrayAttr> padding) {
    auto loc = operand.getLoc();
    const auto opTy = mlir::cast<vpux::NDTypeInterface>(operand.getType());
    auto rank = opTy.getRank();

    if (!padding) {
        return typeConvertedOperand;
    }

    auto shapedTy = mlir::cast<mlir::ShapedType>(operand.getType());
    auto unpaddedShape = SmallVector<int64_t>(shapedTy.getShape());
    if (mlir::failed(IE::unpadInputShape(unpaddedShape, *padding, loc))) {
        return nullptr;
    }
    auto memUnpaddedShape = Shape(DimsOrder::fromValue(operand).toMemoryOrder(ShapeRef(unpaddedShape)).raw());

    SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<mlir::OpFoldResult> sizes(rank);
    for (unsigned i = 0; i < rank; ++i) {
        sizes[i] = rewriter.getIndexAttr(memUnpaddedShape.raw()[i]);
    }

    return rewriter.create<mlir::tensor::ExtractSliceOp>(loc, typeConvertedOperand, offsets, sizes, strides);
}

std::tuple<mlir::Value, mlir::Value> emitTensorSlice(mlir::Location loc, SmallVectorImpl<int64_t>& sliceShape,
                                                     mlir::RankedTensorType allocType,
                                                     mlir::PatternRewriter& rewriter) {
    auto rank = allocType.getRank();
    auto outElemTy = allocType.getElementType();
    mlir::Value zero = nullptr;

    if (mlir::isa<mlir::FloatType>(outElemTy)) {
        zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(outElemTy, 0.));
    } else {
        zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(outElemTy, 0));
    }

    auto outputEmptyTensor = rewriter.create<mlir::tensor::EmptyOp>(loc, allocType.getShape(), outElemTy).getResult();
    auto fillOp =
            rewriter.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{outputEmptyTensor})
                    .result();

    SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> sizes(rank);
    for (unsigned i = 0; i < sliceShape.size(); ++i) {
        sizes[i] = rewriter.getIndexAttr(sliceShape[i]);
    }
    SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto extractOp = rewriter.create<mlir::tensor::ExtractSliceOp>(loc, fillOp, offsets, sizes, strides);

    return {extractOp, fillOp};
}

}  // namespace ShaveCodeGen
}  // namespace vpux
