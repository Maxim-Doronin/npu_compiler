//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZECONCATWITHCONV
#define GEN_PASS_DEF_OPTIMIZECONCATWITHCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// OptimizeConcat
//

class OptimizeConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    OptimizeConcat(mlir::MLIRContext* ctx, int64_t dpuCount, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _dpuCount(dpuCount), _log(log) {
        this->setDebugName("OptimizeConcat");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _dpuCount;
    Logger _log;
};

/*
For subgraph:

        Input0                Input1
    [1, HWC, 1, 1]#NCHW    [1, HWC, 1, 1]#NCHW
            \              /
                 Concat
             [1, H*W*C, 2, 1]#NCHW

Converts to:

        Input0                Input1
    [1, HWC, 1, 1]#NCHW    [1, HWC, 1, 1]#NCHW
          |                      |
        Reshape               Reshape
    [1, C, H, W]#NCHW      [1, C, H, W]#NCHW
          |                     |
       LayoutCast            LayoutCast
    [1, C, H, W]#NHWC    [1, C, H, W]#NHWC
             \                 /
                    Concat
               [1, C, 2H, W]#NHWC
                     |
                    Conv with Kernel[2C, C, H+1, 1]
               [1, 2C, H, W]#NHWC
                     |
                 LayoutCast
               [1, 2C, H, W] #NCHW
                     |
                  Reshape
               [1, H*W*C, 2, 1] #NCHW

*/

mlir::LogicalResult OptimizeConcat::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    auto dimOrder = DimsOrder::fromValue(origOp);
    if (dimOrder != DimsOrder::NCHW) {
        return matchFailed(_log, rewriter, origOp, "Can not apply optimization with layout {0} at '{1}'", dimOrder,
                           origOp->getLoc());
    }
    auto concatAxis = IE::getConcatAxis(origOp);
    if (!concatAxis.has_value() || concatAxis.value().ind() != Dims4D::Act::H.ind()) {
        return matchFailed(_log, rewriter, origOp,
                           "Can not apply optimization with concat axis other than DimH at '{0}'", origOp->getLoc());
    }

    auto concatInputs = origOp.getInputs();
    if (concatInputs.size() != 2) {
        return matchFailed(_log, rewriter, origOp, "Can not apply optimization with concat inputs num {0} at '{1}'",
                           concatInputs.size(), origOp->getLoc());
    }

    auto concatInShape = getShape(concatInputs.front());
    if (concatInShape[Dims4D::Act::N] != 1 || concatInShape[Dims4D::Act::H] != 1 ||
        concatInShape[Dims4D::Act::W] != 1 || concatInShape[Dims4D::Act::C] == 1) {
        return matchFailed(_log, rewriter, origOp, "Cannot apply optimization with concat input shape {0} at '{1}'",
                           concatInShape, origOp->getLoc());
    }

    const auto channelAlignment = VPU::NCEInvariant::getAlignment(origOp.getType());
    const auto H = VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    // Find suitable C which meet SOK requirement, here we limit the search range from [_dpuCount,  _dpuCount*2) to
    // avoid introducing too much workloads
    std::optional<int64_t> suitableC = std::nullopt;
    for (int64_t tileCount = _dpuCount; tileCount < _dpuCount * 2; tileCount++) {
        if (concatInShape.totalSize() % (H * channelAlignment * tileCount) == 0) {
            suitableC = channelAlignment * tileCount;
            break;
        }
    }
    if (!suitableC.has_value()) {
        return matchFailed(
                _log, rewriter, origOp,
                "Can not find suitable DimC size with concat input shape {0} at '{1}', the conversion is skipped",
                concatInShape, origOp->getLoc());
    }
    const auto C = suitableC.value();

    auto areInputsHaveSameShape = llvm::all_of(concatInputs, [&](auto input) {
        return getShape(input) == concatInShape;
    });

    if (!areInputsHaveSameShape) {
        return matchFailed(_log, rewriter, origOp, "concat inputs have different shapes at '{0}'", origOp->getLoc());
    }

    const auto ctx = rewriter.getContext();
    _log.trace("process concat op at {0}", origOp->getLoc());

    // create new concat on outer most dimension
    SmallVector<mlir::Value> inputs;
    Shape newConcatInShape = {1, C, H, concatInShape.totalSize() / (H * C)};
    for (auto input : concatInputs) {
        auto newInReshape = rewriter.create<IE::ReshapeOp>(appendLoc(input.getLoc(), "_reshape"), input, nullptr, false,
                                                           getIntArrayAttr(ctx, newConcatInShape.raw()));
        auto layoutCastOp = rewriter.create<IE::LayoutCastOp>(appendLoc(newInReshape.getLoc(), "_layout_cast"),
                                                              newInReshape, DimsOrder::NHWC.toAffineMap(ctx));
        inputs.push_back(layoutCastOp);
    }
    auto newConcat = rewriter.create<IE::ConcatOp>(origOp.getLoc(), inputs, concatAxis.value());

    // create conv to do the permute
    SmallVector<int64_t> padBegin(2, 0);
    SmallVector<int64_t> padEnd(2, 0);
    SmallVector<int64_t> strides(2, 1);
    SmallVector<int64_t> dilations(2, 1);

    /*
     create weights with shape[2C, C, H+1, 1] for the new concat conv
     The weights values are filled as follows:
                     0                 1          ...        C-1
     Filter0: [1, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
     Filter1: [0, 0,..., 0, 1], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
     Filter2: [0, 0,..., 0, 0], [1, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
     Filter3: [0, 0,..., 0, 0], [0, 0,..., 0, 1], ...,  [0, 0, ..., 0, 0]
     ...
     Filter(2C-2): [0, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [1, 0, ..., 0, 0]
     Filter(2C-1): [0, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 1]
    */
    const auto perFilterSize = H + 1;
    const auto perOutChannelFilterSize = C * perFilterSize;
    const auto OC = 2 * C;
    std::vector<vpux::type::float16> weightsVals(OC * perOutChannelFilterSize, checked_cast<vpux::type::float16>(0.f));
    for (int64_t i = 0; i < C; i++) {
        // for 2i-th filter, set the i-th array first element to 1
        weightsVals[2 * i * perOutChannelFilterSize + i * perFilterSize] = checked_cast<vpux::type::float16>(1.f);
        // for 2i+1-th filter, set the i-th array last element to 1
        weightsVals[(2 * i + 1) * perOutChannelFilterSize + i * perFilterSize + H] =
                checked_cast<vpux::type::float16>(1.f);
    }

    Shape weightsShape = {2 * C, C, H + 1, 1};
    const auto weightStorageType = mlir::RankedTensorType::get(weightsShape.raw(), mlir::Float16Type::get(ctx));
    const auto weightStorageAttr = Const::createConstContent(weightStorageType, ArrayRef(weightsVals));
    const auto weightContentAttr = Const::ContentAttr::get(weightStorageAttr);
    const auto declLoc = appendLoc(origOp.getLoc(), "weights_for_concat");

    const auto weightExpressedElemType =
            IE::composeWeightsExpressedType(newConcat.getResult().getType().getElementType());
    const auto weightExpressedType = mlir::RankedTensorType::get(weightsShape.raw(), weightExpressedElemType);
    auto targetContentAttr = weightContentAttr.transform().castElemType(weightExpressedElemType).get();
    auto weightsConst = rewriter.create<Const::DeclareOp>(declLoc, weightExpressedType, std::move(targetContentAttr));
    const auto reorderLoc = appendLoc(weightsConst.getLoc(), "reorder_weights_for_DPU_concat");
    const auto weightTypeNCHW = mlir::cast<vpux::NDTypeInterface>(weightsConst.getOutput().getType());
    const auto reorderType = weightTypeNCHW.changeDimsOrder(DimsOrder::NHWC);
    const auto orderMap = DimsOrder::NHWC.toAffineMap(ctx);
    auto weightsReorder =
            rewriter.createOrFold<IE::ReorderOp>(reorderLoc, reorderType, weightsConst.getOutput(), orderMap);

    auto newConv = rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), newConcat, weightsReorder,
                                                      /*bias=*/nullptr, getIntArrayAttr(ctx, strides),
                                                      getIntArrayAttr(ctx, padBegin), getIntArrayAttr(ctx, padEnd),
                                                      getIntArrayAttr(ctx, dilations),
                                                      /*postOp=*/nullptr, /*clamp=*/nullptr, /*staticScale=*/nullptr,
                                                      /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
    _log.trace("create new conv {0}", newConv);
    auto newOutLayoutCast = rewriter.create<IE::LayoutCastOp>(appendLoc(newConv.getLoc(), "_layout_cast_"),
                                                              newConv.getOutput(), DimsOrder::NCHW.toAffineMap(ctx));

    auto concatOutShape = getShape(origOp);
    auto newOutReshape =
            rewriter.create<IE::ReshapeOp>(appendLoc(newOutLayoutCast.getLoc(), "_reshape_"), newOutLayoutCast, nullptr,
                                           false, getIntArrayAttr(ctx, concatOutShape.raw()));
    rewriter.replaceAllUsesWith(origOp, newOutReshape);
    return mlir::success();
}

//
// OptimizeConcatWithConvAndAdd
//

class OptimizeConcatWithConvAndAdd final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    OptimizeConcatWithConvAndAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        this->setDebugName("OptimizeConcatWithConvAndAdd");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const final;
    bool isBeneficialToConvert(IE::ConcatOp concatOp) const;

private:
    Logger _log;
};

/*
For example:
        Input0                Input1
    [1, C, H, W]#NHWC    [1, C, H, W]#NHWC
            \              /
                 Concat
             [1, 2C, H, W]#NHWC
Converts to:
   Filter1          Input0                Input1         Filter2
[2C, C, 1, 1]    [1, C, H, W]#NHWC    [1, C, H, W]#NHWC [2C, C, 1, 1]
        \              |                      |        /
                Convolution1           Convolution2
                [1, 2C, H, W]#NHWC      [1, 2C, H, W]#NHWC
                       |                     |
                        \                   /
                                Add
                            [1, 2C, H, W]#NHWC
Filter1 : [1], [0], ..., [0]
Filter2 : [0], [1], ..., [0]
...
Filter2C: [0], [0], ..., [1]
*/
mlir::LogicalResult OptimizeConcatWithConvAndAdd::matchAndRewrite(IE::ConcatOp concatOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!isBeneficialToConvert(concatOp)) {
        return matchFailed(_log, rewriter, concatOp, "Not beneficial to convert concat at '{0}'", concatOp->getLoc());
    }

    // Compose weights for the convolution, the new weights output channels are the same as the concat output channels
    // and those channels value are 1. And other channels are 0. And than those convolution ops' outputs can be added
    // together to get the final concat results.
    auto composeWeights = [](int inChannels, int outChannels, int startChannel) {
        SmallVector<float> weightsVals(inChannels * outChannels, checked_cast<vpux::type::float16>(0.f));
        for (int i = 0; i < inChannels; ++i) {
            auto targetChannel = startChannel + i;
            auto weightIndex = targetChannel * inChannels + i;
            weightsVals[weightIndex] = 1.0f;
        }
        return weightsVals;
    };

    auto ctx = rewriter.getContext();

    auto concatInputs = concatOp.getInputs();
    auto concatOutput = concatOp.getOutput();
    auto concatOutShape = getShape(concatOutput);
    auto totalOutChannels = concatOutShape[Dims4D::Act::C];
    auto accumulateChannels = 0;

    SmallVector<mlir::Value> convResults;
    SmallVector<mlir::Operation*> constVec;
    SmallVector<mlir::Operation*> convVec;

    for (size_t i = 0; i < concatInputs.size(); ++i) {
        auto currentInput = concatInputs[i];
        auto currentType = mlir::cast<vpux::NDTypeInterface>(currentInput.getType());
        const int64_t currentInChannels = currentType.getShape()[Dims4D::Act::C];

        auto weightsVals = composeWeights(currentInChannels, totalOutChannels, accumulateChannels);
        accumulateChannels += currentInChannels;

        const Shape weightsShape = {totalOutChannels, currentInChannels, 1, 1};
        const DimsOrder weightOrder = DimsOrder::OYXI;

        VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(currentInput.getType()),
                          "{0} doesn't support dynamic shapes", IE::ConvolutionOp::getOperationName());
        const auto weightType = mlir::RankedTensorType::get(
                weightsShape.raw(), mlir::cast<NDTypeInterface>(currentInput.getType()).getElementType(),
                getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
        auto weight = Const::buildWeightsConst(rewriter, currentInput.getLoc(), weightType, ArrayRef(weightsVals));
        constVec.push_back(weight.getDefiningOp());

        const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
        const auto kernelPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        const auto kernelPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

        const auto convLoc = takeOpLoc(concatOp, llvm::formatv("convolution_for_concat_{0}", i).str());
        auto newConv =
                rewriter.create<IE::ConvolutionOp>(convLoc, currentInput, weight,
                                                   /*bias=*/nullptr, strides, kernelPadsBegin, kernelPadsEnd, dilations,
                                                   /*postOp=*/nullptr, /*clamp=*/nullptr, /*staticScale=*/nullptr,
                                                   /*outputChannels=*/nullptr, /*inputChannels=*/nullptr);
        convVec.push_back(newConv);
        const auto adjustConvShapeParameters = getAdjustConvShapeParameters(newConv, weight, concatOutShape, _log);
        if (mlir::failed(adjustConvShapeParameters)) {
            // If the conv shape is not adjusted successfully, skip the conversion, we need to erase the conv and weight
            // ops
            for (auto conv : convVec) {
                rewriter.eraseOp(conv);
            }
            for (auto weight : constVec) {
                rewriter.eraseOp(weight);
            }
            return mlir::failure();
        }

        convResults.push_back(newConv.getOutput());
    }

    IE::AddOp addOp;
    for (size_t i = 1; i < convResults.size(); ++i) {
        const auto declLoc = takeOpLoc(concatOp, llvm::formatv("add_for_concat_{0}", i).str());
        addOp = rewriter.create<IE::AddOp>(declLoc, convResults[0], convResults[i],
                                           IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NUMPY),
                                           nullptr, nullptr, nullptr, nullptr);
        convResults[0] = addOp.getOutput();
    }

    rewriter.replaceAllUsesWith(concatOp, addOp);
    return mlir::success();
}

bool OptimizeConcatWithConvAndAdd::isBeneficialToConvert(IE::ConcatOp concatOp) const {
    auto dimOrder = DimsOrder::fromValue(concatOp);
    if (dimOrder != DimsOrder::NHWC) {
        _log.trace("Concat input at {0} layout not support", concatOp.getLoc());
        return false;
    }

    auto concatAxis = IE::getConcatAxis(concatOp);
    if (!concatAxis.has_value() || concatAxis.value().ind() != Dims4D::Act::C.ind()) {
        _log.trace("Concat input at {0} concat axis is not channel", concatOp.getLoc());
        return false;
    }

    // Experimental number to determine if the concat input is efficient to convert. Details data: E161332.
    constexpr int CONVERT_CONCAT_RATIO = 10000;
    auto concatOutput = concatOp.getOutput();
    auto concatOutShape = getShape(concatOutput);
    if ((concatOutShape[Dims4D::Act::H] * concatOutShape[Dims4D::Act::W]) / concatOutShape[Dims4D::Act::C] <
        CONVERT_CONCAT_RATIO) {
        _log.trace("Concat input at {0} not efficient to convert", concatOp.getLoc());
        return false;
    }

    return true;
}

//
// OptimizeConcatWithConvPass
//

class OptimizeConcatWithConvPass final : public IE::impl::OptimizeConcatWithConvBase<OptimizeConcatWithConvPass> {
public:
    explicit OptimizeConcatWithConvPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void OptimizeConcatWithConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto moduleOp = func->getParentOfType<mlir::ModuleOp>();
    auto tileOp = IE::getTileExecutor(moduleOp);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeConcat>(&ctx, tileOp.getCount(), _log);
    patterns.add<OptimizeConcatWithConvAndAdd>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeConcatWithConvPass(Logger log) {
    return std::make_unique<OptimizeConcatWithConvPass>(log);
}
