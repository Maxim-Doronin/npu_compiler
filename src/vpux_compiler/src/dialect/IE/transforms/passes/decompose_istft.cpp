//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEISTFT
#define GEN_PASS_DEF_DECOMPOSEISTFT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Helper function to create a slice operation for frame extraction
IE::SliceOp createFrameSlice(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                             ArrayRef<int64_t> inputShape, int64_t frameIdx, mlir::MLIRContext* ctx) {
    SmallVector<int64_t> offsets(inputShape.size(), 0);
    SmallVector<int64_t> sizes = to_small_vector(inputShape);

    offsets[offsets.size() - 2] = frameIdx;
    sizes[sizes.size() - 2] = 1;

    return rewriter.create<IE::SliceOp>(appendLoc(loc, "frame_{0}_slice", frameIdx), input,
                                        getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes));
}

//
// ISTFTOpConverter
//

class ISTFTOpConverter final : public mlir::OpRewritePattern<IE::ISTFTOp> {
public:
    ISTFTOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ISTFTOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ISTFTOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ISTFTOpConverter::matchAndRewrite(IE::ISTFTOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto ctx = origOp.getContext();
    const auto loc = origOp.getLoc();

    _log.trace("Decomposing ISTFT operation at {0}", loc);

    auto signal = origOp.getSignal();
    auto window = origOp.getWindow();
    auto frameSize = origOp.getFrameSize();
    auto frameStep = origOp.getFrameStep();
    auto signalLengthInput = origOp.getSignalLength();
    const auto center = origOp.getCenter().has_value();
    const auto normalized = origOp.getNormalized().has_value();

    auto frameSizeConstOp = frameSize.getDefiningOp<Const::DeclareOp>();
    auto frameStepConstOp = frameStep.getDefiningOp<Const::DeclareOp>();

    if (!frameSizeConstOp || !frameStepConstOp) {
        _log.error("ISTFT frame_size and frame_step must be constants for compile-time unrolling");
        return mlir::failure();
    }

    const auto frameSizeContent = frameSizeConstOp.getContent();
    const auto frameStepContent = frameStepConstOp.getContent();
    if (!frameSizeContent.isSplat() || !frameStepContent.isSplat()) {
        return mlir::failure();
    }
    const auto frameSizeVal = frameSizeContent.getSplatValue<int64_t>();
    const auto frameStepVal = frameStepContent.getSplatValue<int64_t>();

    const auto signalType = mlir::cast<vpux::NDTypeInterface>(signal.getType());
    const auto signalShape = signalType.getShape();
    const auto elemType = signalType.getElementType();

    if (signalShape.size() < 3) {
        _log.error("ISTFT input must have at least 3 dimensions, got {0}D", signalShape.size());
        return mlir::failure();
    }

    const auto signalShapeRaw = signalShape.raw();
    const auto fftSize = signalShapeRaw[signalShapeRaw.size() - 3];
    const auto numFrames = signalShapeRaw[signalShapeRaw.size() - 2];

    const auto expectedFftSize = frameSizeVal / 2 + 1;
    if (fftSize != expectedFftSize) {
        _log.error("ISTFT frequency bins {0} don't match expected {1} for frameSize {2}", fftSize, expectedFftSize,
                   frameSizeVal);
        return mlir::failure();
    }

    if (frameSizeVal <= 0 || frameStepVal <= 0) {
        _log.error("Frame size {0} and step {1} must be positive", frameSizeVal, frameStepVal);
        return mlir::failure();
    }

    int64_t signalLength;
    if (signalLengthInput) {
        auto signalLengthConstOp = signalLengthInput.getDefiningOp<Const::DeclareOp>();
        if (!signalLengthConstOp) {
            _log.error("ISTFT signal_length must be a constant");
            return mlir::failure();
        }
        const auto signalLengthContent = signalLengthConstOp.getContent();
        if (!signalLengthContent.isSplat()) {
            return mlir::failure();
        }
        signalLength = signalLengthContent.getSplatValue<int64_t>();
    } else {
        // Calculate default signal_length based on center attribute
        if (center) {
            // center=true: default_signal_length = (frames - 1) * frame_step
            signalLength = (numFrames - 1) * frameStepVal;
        } else {
            // center=false: default_signal_length = (frames - 1) * frame_step + frame_size
            signalLength = (numFrames - 1) * frameStepVal + frameSizeVal;
        }
    }

    SmallVector<int64_t> outputShape;
    for (size_t i = 0; i < signalShapeRaw.size() - 3; ++i) {
        outputShape.push_back(signalShapeRaw[i]);
    }
    outputShape.push_back(signalLength);

    SmallVector<mlir::Value> frames;
    frames.reserve(numFrames);

    SmallVector<int64_t> inputSignalShape = to_small_vector(signalShape.raw());

    for (int64_t frameIdx = 0; frameIdx < numFrames; frameIdx++) {
        auto sliceOp = createFrameSlice(rewriter, loc, signal, inputSignalShape, frameIdx, ctx);
        frames.push_back(sliceOp.getResult());
    }

    SmallVector<mlir::Value> reshapedFrames;
    reshapedFrames.reserve(numFrames);

    for (auto frame : frames) {
        SmallVector<int64_t> frameWithAxisShape;
        for (size_t i = 0; i < signalShape.size() - 3; ++i) {
            frameWithAxisShape.push_back(signalShape.raw()[i]);
        }
        frameWithAxisShape.push_back(fftSize);
        frameWithAxisShape.push_back(2);

        const auto frameWithAxisType = mlir::RankedTensorType::get(frameWithAxisShape, elemType);
        const auto frameWithAxisShapeAttr = getIntArrayAttr(ctx, frameWithAxisShape);

        auto frameReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "frame_reshape"), frameWithAxisType, frame,
                                                             frameWithAxisShapeAttr);
        reshapedFrames.push_back(frameReshapeOp.getOutput());
    }

    SmallVector<int64_t> frameWithAxisShape;
    for (size_t i = 0; i < signalShape.size() - 3; ++i) {
        frameWithAxisShape.push_back(signalShape.raw()[i]);
    }
    frameWithAxisShape.push_back(fftSize);
    frameWithAxisShape.push_back(2);

    SmallVector<int64_t> irdftAxes = {static_cast<int64_t>(frameWithAxisShape.size() - 2)};
    SmallVector<int64_t> irdftSignalSize = {frameSizeVal};

    const auto irdftAxesAttr = getIntArrayAttr(ctx, irdftAxes);
    const auto irdftSignalSizeAttr = getIntArrayAttr(ctx, irdftSignalSize);

    SmallVector<int64_t> irdftOutputShape;
    for (size_t i = 0; i < signalShape.size() - 3; ++i) {
        irdftOutputShape.push_back(signalShape.raw()[i]);
    }
    irdftOutputShape.push_back(frameSizeVal);

    const auto irdftOutputType = mlir::RankedTensorType::get(irdftOutputShape, elemType);

    SmallVector<mlir::Value> windowedFrames;
    windowedFrames.reserve(numFrames);

    for (int64_t frameIdx = 0; frameIdx < numFrames; frameIdx++) {
        auto reshapedFrame = reshapedFrames[frameIdx];

        auto irdftOp =
                rewriter.create<IE::IRDFTOp>(appendLoc(loc, "frame_{0}_irdft", frameIdx), irdftOutputType,
                                             reshapedFrame, nullptr, nullptr, irdftAxesAttr, irdftSignalSizeAttr);
        mlir::Value windowedFrame = irdftOp.getResult();
        if (window) {
            auto autoBroadcastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
            windowedFrame = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "frame_{0}_windowing_multiply", frameIdx),
                                                            windowedFrame, window, autoBroadcastAttr, nullptr, nullptr,
                                                            nullptr, nullptr)
                                    .getOutput();
        }
        if (normalized) {
            const auto normFactor = std::sqrt(static_cast<double>(frameSizeVal));
            const auto scalarType = mlir::RankedTensorType::get({}, elemType);
            const auto normFactorValue = static_cast<float>(normFactor);
            auto normFactorConst = Const::createConst(rewriter, loc, scalarType, ArrayRef<float>{normFactorValue});

            auto autoBroadcastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
            windowedFrame = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "frame_{0}_normalize", frameIdx),
                                                            windowedFrame, normFactorConst, autoBroadcastAttr, nullptr,
                                                            nullptr, nullptr, nullptr)
                                    .getOutput();
        }

        windowedFrames.push_back(windowedFrame);
    }

    auto outputType = mlir::RankedTensorType::get(outputShape, elemType);
    mlir::Value finalResult = Const::createZerosConst(rewriter, loc, outputType);
    mlir::Value windowSum = Const::createZerosConst(rewriter, loc, outputType);

    for (int64_t frameIdx = 0; frameIdx < numFrames; frameIdx++) {
        auto windowedFrame = windowedFrames[frameIdx];
        int64_t frameStart;
        if (center) {
            frameStart = static_cast<int64_t>(frameIdx) * static_cast<int64_t>(frameStepVal) -
                         static_cast<int64_t>(frameSizeVal) / 2;
        } else {
            frameStart = static_cast<int64_t>(frameIdx) * static_cast<int64_t>(frameStepVal);
        }

        int64_t frameEnd = frameStart + static_cast<int64_t>(frameSizeVal);
        int64_t outputStart = std::max(frameStart, static_cast<int64_t>(0));
        int64_t outputEnd = std::min(frameEnd, static_cast<int64_t>(signalLength));

        if (outputStart >= outputEnd) {
            continue;
        }

        int64_t frameSliceStart = outputStart - frameStart;
        int64_t frameSliceSize = outputEnd - outputStart;

        if (frameSliceStart > 0 || frameSliceSize < frameSizeVal) {
            SmallVector<int64_t> sliceOffsets(mlir::cast<mlir::ShapedType>(windowedFrame.getType()).getRank(), 0);
            SmallVector<int64_t> sliceSizes =
                    to_small_vector(mlir::cast<mlir::ShapedType>(windowedFrame.getType()).getShape());

            sliceOffsets.back() = frameSliceStart;
            sliceSizes.back() = frameSliceSize;

            windowedFrame =
                    rewriter.create<IE::SliceOp>(appendLoc(loc, "frame_{0}_truncate", frameIdx), windowedFrame,
                                                 getIntArrayAttr(ctx, sliceOffsets), getIntArrayAttr(ctx, sliceSizes))
                            .getResult();
        }

        const auto padBefore = outputStart;
        const auto padAfter = signalLength - outputEnd;

        SmallVector<int64_t> padBeforeVec(mlir::cast<mlir::ShapedType>(windowedFrame.getType()).getRank(), 0);
        SmallVector<int64_t> padAfterVec(mlir::cast<mlir::ShapedType>(windowedFrame.getType()).getRank(), 0);
        padBeforeVec.back() = padBefore;
        padAfterVec.back() = padAfter;

        auto padModeAttr = IE::PadModeAttr::get(ctx, IE::PadMode::CONSTANT);
        auto paddedFrame = rewriter.create<IE::PadOp>(appendLoc(loc, "frame_{0}_pad", frameIdx), windowedFrame, nullptr,
                                                      nullptr, nullptr, getIntArrayAttr(ctx, padBeforeVec),
                                                      getIntArrayAttr(ctx, padAfterVec), getFPAttr(ctx, 0.0),
                                                      padModeAttr, nullptr, nullptr, nullptr, nullptr);

        auto autoBroadcastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
        finalResult = rewriter.create<IE::AddOp>(appendLoc(loc, "add_frame_{0}", frameIdx), finalResult,
                                                 paddedFrame.getResult(), autoBroadcastAttr, nullptr, nullptr, nullptr,
                                                 nullptr)
                              .getOutput();

        if (window) {
            mlir::Value windowForSum = window;
            if (frameSliceStart > 0 || frameSliceSize < frameSizeVal) {
                SmallVector<int64_t> sliceOffsets(1, frameSliceStart);
                SmallVector<int64_t> sliceSizes(1, frameSliceSize);

                windowForSum = rewriter.create<IE::SliceOp>(appendLoc(loc, "window_{0}_crop", frameIdx), window,
                                                            getIntArrayAttr(ctx, sliceOffsets),
                                                            getIntArrayAttr(ctx, sliceSizes))
                                       .getResult();
            }

            auto powWindow =
                    rewriter.create<IE::MultiplyOp>(appendLoc(loc, "window_pow_{0}", frameIdx), windowForSum,
                                                    windowForSum, autoBroadcastAttr, nullptr, nullptr, nullptr, nullptr)
                            .getOutput();

            SmallVector<int64_t> winBeforeVec(1, outputStart);
            SmallVector<int64_t> winAfterVec(1, signalLength - outputEnd);

            auto padModeAttr = IE::PadModeAttr::get(ctx, IE::PadMode::CONSTANT);
            auto powWindowPadded = rewriter.create<IE::PadOp>(
                    appendLoc(loc, "window_pad_{0}", frameIdx), powWindow, nullptr, nullptr, nullptr,
                    getIntArrayAttr(ctx, winBeforeVec), getIntArrayAttr(ctx, winAfterVec), getFPAttr(ctx, 0.0),
                    padModeAttr, nullptr, nullptr, nullptr, nullptr);

            windowSum = rewriter.create<IE::AddOp>(appendLoc(loc, "add_window_sum_{0}", frameIdx), windowSum,
                                                   powWindowPadded.getResult(), autoBroadcastAttr, nullptr, nullptr,
                                                   nullptr, nullptr)
                                .getOutput();
        }
    }

    if (window && windowSum) {
        float epsilonValue = 1e-4f;
        const auto scalarType = mlir::RankedTensorType::get({}, elemType);
        auto epsilonConst = Const::createConst(rewriter, loc, scalarType, ArrayRef<float>{epsilonValue});
        auto autoBroadcastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
        windowSum = rewriter.create<IE::AddOp>(appendLoc(loc, "window_sum_epsilon"), windowSum, epsilonConst,
                                               autoBroadcastAttr, nullptr, nullptr, nullptr, nullptr)
                            .getOutput();

        finalResult = rewriter.create<IE::DivideOp>(appendLoc(loc, "normalize_by_window"), finalResult, windowSum,
                                                    autoBroadcastAttr)
                              .getOutput();
    }

    rewriter.replaceOp(origOp, finalResult);

    _log.trace("Successfully decomposed ISTFT operation with {0} frames", numFrames);
    return mlir::success();
}

//
// DecomposeISTFTPass
//

class DecomposeISTFTPass final : public IE::impl::DecomposeISTFTBase<DecomposeISTFTPass> {
public:
    explicit DecomposeISTFTPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DecomposeISTFTPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::ISTFTOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::IRDFTOp>();
    target.addLegalOp<IE::MultiplyOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<IE::DivideOp>();
    target.addLegalOp<IE::PadOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ISTFTOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeISTFTPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeISTFTPass(Logger log) {
    return std::make_unique<DecomposeISTFTPass>(log);
}
