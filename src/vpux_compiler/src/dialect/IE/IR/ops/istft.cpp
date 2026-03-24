//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<int64_t> extractConstantParam(mlir::Value param, const char* paramName, mlir::Location loc) {
    if (!param) {
        return errorAt(loc, "ISTFT {0} must be specified", paramName);
    }

    auto constOp = mlir::dyn_cast_or_null<vpux::Const::DeclareOp>(param.getDefiningOp());
    if (!constOp) {
        return errorAt(loc, "ISTFT {0} must be a constant", paramName);
    }

    const auto content = constOp.getContent();
    if (!content.isSplat()) {
        return errorAt(loc, "ISTFT {0} must be a splat", paramName);
    }

    return content.getSplatValue<int64_t>();
}

}  // namespace

mlir::LogicalResult vpux::IE::ISTFTOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::ISTFTOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto signalType = mlir::cast<mlir::ShapedType>(op.getSignal().getType());
    const auto signalShape = signalType.getShape();
    const auto elemType = signalType.getElementType();

    auto frameSizeOrError = extractConstantParam(op.getFrameSize(), "frame_size", loc);
    auto frameStepOrError = extractConstantParam(op.getFrameStep(), "frame_step", loc);

    if (mlir::failed(frameSizeOrError) || mlir::failed(frameStepOrError)) {
        return mlir::failure();
    }

    const auto frameSize = frameSizeOrError.value();
    const auto frameStep = frameStepOrError.value();

    const auto center = op.getCenter().has_value() ? op.getCenter().value() : false;

    auto outputShape = to_small_vector(signalShape);

    const auto fftSize = signalShape[signalShape.size() - 3];
    const auto numFrames = signalShape[signalShape.size() - 2];

    const auto expectedFftSize = frameSize / 2 + 1;
    if (fftSize != expectedFftSize) {
        return errorAt(loc, "ISTFT frequency bins {0} don't match expected {1} for frameSize {2}", fftSize,
                       expectedFftSize, frameSize);
    }

    int64_t signalLength;
    if (op.getSignalLength()) {
        auto signalLengthOrError = extractConstantParam(op.getSignalLength(), "signal_length", loc);
        if (mlir::failed(signalLengthOrError)) {
            return mlir::failure();
        }
        signalLength = signalLengthOrError.value();
    } else {
        if (center) {
            signalLength = (numFrames - 1) * frameStep;
        } else {
            signalLength = (numFrames - 1) * frameStep + frameSize;
        }
    }

    outputShape.resize(signalShape.size() - 3);
    outputShape.push_back(signalLength);

    inferredReturnShapes.emplace_back(outputShape, elemType);
    return mlir::success();
}

mlir::LogicalResult vpux::IE::ISTFTOp::verify() {
    const auto signalType = mlir::cast<mlir::ShapedType>(getSignal().getType());
    const auto signalShape = signalType.getShape();

    if (signalShape.size() < 3 || signalShape.size() > 4) {
        return errorAt(getLoc(), "ISTFT signal must have 3 or 4 dimensions, got {0}D", signalShape.size());
    }

    if (signalShape.back() != 2) {
        return errorAt(getLoc(), "ISTFT signal last dimension must be 2 (complex value), got {0}", signalShape.back());
    }

    auto frameSizeOrError = extractConstantParam(getFrameSize(), "frame_size", getLoc());
    auto frameStepOrError = extractConstantParam(getFrameStep(), "frame_step", getLoc());

    if (mlir::failed(frameSizeOrError) || mlir::failed(frameStepOrError)) {
        return mlir::failure();
    }

    const auto frameSize = frameSizeOrError.value();
    const auto frameStep = frameStepOrError.value();

    if (frameSize <= 0) {
        return errorAt(getLoc(), "ISTFT frame_size must be positive, got {0}", frameSize);
    }

    if (frameStep <= 0) {
        return errorAt(getLoc(), "ISTFT frame_step must be positive, got {0}", frameStep);
    }

    if (getWindow()) {
        const auto windowType = mlir::cast<mlir::ShapedType>(getWindow().getType());
        const auto windowShape = windowType.getShape();

        if (windowShape.size() != 1) {
            return errorAt(getLoc(), "ISTFT window must be 1D tensor, got {0}D", windowShape.size());
        }
    }

    return mlir::success();
}
