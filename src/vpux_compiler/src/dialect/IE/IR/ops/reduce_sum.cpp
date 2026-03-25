//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

void IE::ReduceSumOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Type outputType,
                            mlir::Value input, mlir::Value axes, mlir::ArrayAttr axesValue, mlir::UnitAttr keepDims) {
    return build(odsBuilder, odsState, outputType, input, axes, axesValue, keepDims, /*outputPadding=*/nullptr,
                 /*inputPadding=*/nullptr);
}

void IE::ReduceSumOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                            mlir::Value axes, mlir::ArrayAttr axesValue, mlir::UnitAttr keepDims) {
    return build(odsBuilder, odsState, input, axes, axesValue, keepDims, /*outputPadding=*/nullptr,
                 /*inputPadding=*/nullptr);
}

mlir::LogicalResult vpux::IE::ReduceSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceSumOpAdaptor reduceSum(operands, attrs, prop);
    if (mlir::failed(reduceSum.verify(loc))) {
        return mlir::failure();
    }
    if (reduceSum.getAxes() != nullptr && reduceSum.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceSum.getAxes() == nullptr && !reduceSum.getAxesValue().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceSum.getInput();
    const auto keepDims = reduceSum.getKeepDims();

    auto axesValue = IE::extractAxes(loc, reduceSum);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes,
                                               reduceSum.getInputPaddingAttr(), reduceSum.getOutputPaddingAttr());
}

mlir::LogicalResult vpux::IE::ReduceSumOp::verify() {
    llvm::SmallVector<int64_t> axesVec;
    const auto op = getOperation();
    if (getAxes() != nullptr) {
        const auto opAxes = mlir::dyn_cast<mlir::RankedTensorType>(getAxes().getType());

        if (opAxes == nullptr) {
            return errorAt(op, "Axes is not a 'RankedTensorType', got '{0}'", opAxes);
        }

        const auto axesRank = opAxes.getRank();

        // The axes input must be a scalar or 1D tensor
        if (axesRank > 1) {
            return errorAt(
                    op, "Operation has unsupported tensor rank '{0}' for axes, it must be either a scalar or 1D tensor",
                    axesRank);
        }
        // The axes input must have integer type.
        if (!mlir::isa<mlir::IntegerType>(opAxes.getElementType())) {
            return errorAt(op, " Axes input must have integer element type but actual element type is '{0}'",
                           opAxes.getElementType());
        }

        // The axes input must contain unique elements
        axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(getAxes()));
    }

    if (getAxesValue().has_value()) {
        const auto opAxesValue = getAxesValue().value();

        // The axes input must contain unique elements
        axesVec = parseIntArrayAttr<int64_t>(opAxesValue);
    }

    llvm::sort(axesVec);
    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(op, "Axes values should be unique");
    }

    if (mlir::failed(IE::checkPadding(getInputPaddingAttr(), getInput().getType()))) {
        return errorAt(op, "Input padding {0} incompatible with input type {1}", getInputPaddingAttr(),
                       getInput().getType());
    }
    if (mlir::failed(IE::checkPadding(getOutputPaddingAttr(), getOutput().getType()))) {
        return errorAt(op, "Output padding {0} incompatible with output type {1}", getOutputPaddingAttr(),
                       getOutput().getType());
    }

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceSumOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        if (getInputPaddingAttr() == nullptr && getOutputPaddingAttr() == nullptr) {
            return getInput();
        }

        // In case the operation has padding, check if the non-padded shapes are the same. If they are, the operation
        // can be folded as there is nothing to reduce on the given axes
        auto inputShape = SmallVector<int64_t>(mlir::cast<NDTypeInterface>(getInput().getType()).getShape().raw());
        if (getInputPaddingAttr() != nullptr) {
            auto inputPadding = parseIntArrayAttr<int64_t>(getInputPaddingAttr());
            for (size_t i = 0; i < inputShape.size(); ++i) {
                inputShape[i] -= inputPadding[i];
            }
        }
        auto outputShape = SmallVector<int64_t>(mlir::cast<NDTypeInterface>(getOutput().getType()).getShape().raw());
        if (getOutputPaddingAttr() != nullptr) {
            auto outputPadding = parseIntArrayAttr<int64_t>(getOutputPaddingAttr());
            for (size_t i = 0; i < outputShape.size(); ++i) {
                outputShape[i] -= outputPadding[i];
            }
        }
        if (inputShape == outputShape) {
            return getInput();
        }
    }

    return nullptr;
}

void vpux::IE::ReduceSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr<vpux::IE::ReduceSumOp>>(context);
}
