//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEQUANTIZATIONMULTIPLY
#define GEN_PASS_DEF_FUSEQUANTIZATIONMULTIPLY
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

SmallVector<float> getConstContentVals(vpux::Const::Content constContent) {
    return constContent.isSplat() ? SmallVector<float>{constContent.getSplatValue<float>()}
                                  : SmallVector<float>(constContent.getValues<float>());
}

//
// FuseFQAndMul
//

class FuseFQAndMul final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    FuseFQAndMul(mlir::MLIRContext* ctx, Logger log, bool fuseFQAndMulWithNonConstInput)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx),
              _log(log),
              _fuseFQAndMulWithNonConstInput(fuseFQAndMulWithNonConstInput) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalToFuse(IE::MultiplyOp multiplyOp) const;

private:
    Logger _log;
    bool _fuseFQAndMulWithNonConstInput = false;
};

bool FuseFQAndMul::isLegalToFuse(IE::MultiplyOp multiplyOp) const {
    bool lhsIsActivation = vpux::VPU::isEltwiseLhsActivation<IE::MultiplyOp>(multiplyOp);

    auto fakeQuantOp = lhsIsActivation ? multiplyOp.getInput1().getDefiningOp<IE::FakeQuantizeOp>()
                                       : multiplyOp.getInput2().getDefiningOp<IE::FakeQuantizeOp>();
    if (fakeQuantOp == nullptr || !fakeQuantOp->hasOneUse()) {
        return false;
    }

    auto constInputOp = fakeQuantOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (constInputOp == nullptr && !_fuseFQAndMulWithNonConstInput) {
        return false;
    }

    auto mulConstOp = lhsIsActivation ? multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>()
                                      : multiplyOp.getInput1().getDefiningOp<Const::DeclareOp>();

    auto outLowConst = fakeQuantOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fakeQuantOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (mulConstOp == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return false;
    }

    auto outLowConstShape = getShape(outLowConst.getOutput());
    auto outHighConstShape = getShape(outHighConst.getOutput());
    if (outLowConstShape != outHighConstShape) {
        return false;
    }

    const auto mulConstContent = mulConstOp.getContent();
    if (mulConstContent.isSplat()) {
        return true;
    }

    auto mulConstShape = getShape(mulConstOp.getOutput());
    const auto mulNoneOneAxisCount = std::count_if(mulConstShape.begin(), mulConstShape.end(), [](auto size) {
        return size > 1;
    });

    return (mulNoneOneAxisCount == 1) && (outHighConstShape.totalSize() == 1 || mulConstShape == outHighConstShape);
}

/*
      data  in_L in_H out_L out_H
        |    |    |     |     |
        |    |    |     |     |                data  in_L in_H  out_L * C  out_H * C
        v    v    v     v     v                  |    |    |        |          |
      +-------------------------+                |    |    |        |          |
      |       FakeQuantize      |                v    v    v        v          v
      +-------------------------+             +-----------------------------------+
                   |                =====>    |            FakeQuantize           |
                   v                          +-----------------------------------+
              +----------+                                      |
              | Multiply | <--- C                               v
              +----+-----+
                   |
                   v
*/

mlir::LogicalResult FuseFQAndMul::matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const {
    if (!isLegalToFuse(multiplyOp)) {
        return mlir::failure();
    }

    bool lhsIsActivation = vpux::VPU::isEltwiseLhsActivation<IE::MultiplyOp>(multiplyOp);

    auto fakeQuantOp = lhsIsActivation ? multiplyOp.getInput1().getDefiningOp<IE::FakeQuantizeOp>()
                                       : multiplyOp.getInput2().getDefiningOp<IE::FakeQuantizeOp>();
    auto mulConstOp = lhsIsActivation ? multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>()
                                      : multiplyOp.getInput1().getDefiningOp<Const::DeclareOp>();

    auto outLowConst = fakeQuantOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fakeQuantOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    _log.trace("Fuse Mul '{0}' into FQ '{1}'", multiplyOp->getLoc(), fakeQuantOp->getLoc());

    auto mulConstShape = Shape(getShape(mulConstOp.getOutput()));
    auto outType = mlir::cast<NDTypeInterface>(outHighConst.getType());
    auto newOutType = mulConstOp.getContent().isSplat() ? outType : outType.changeShape(mulConstShape);
    const auto newOutRankedTensorType = mlir::cast<mlir::RankedTensorType>(newOutType);

    Const::ContentAttr newOutLowContentAttr = nullptr;
    Const::ContentAttr newOutHighContentAttr = nullptr;
    auto mulConstVals = getConstContentVals(mulConstOp.getContent());
    if (mulConstVals.size() == 1) {
        const double scaleVal = mulConstVals.front();
        newOutLowContentAttr = outLowConst.transformContentAttr().rescale(scaleVal).get();
        newOutHighContentAttr = outHighConst.transformContentAttr().rescale(scaleVal).get();
    } else {
        auto outLowVals = getConstContentVals(outLowConst.getContent());
        auto outHighVals = getConstContentVals(outHighConst.getContent());

        auto fqOutValsCount = std::max(mulConstVals.size(), outLowVals.size());
        SmallVector<float> newOutHighVals(fqOutValsCount);
        SmallVector<float> newOutLowVals(fqOutValsCount);

        for (size_t idx = 0; idx < fqOutValsCount; ++idx) {
            auto outHighVal = outHighVals.size() == 1 ? outHighVals.front() : outHighVals[idx];
            auto outLowVal = outLowVals.size() == 1 ? outLowVals.front() : outLowVals[idx];
            auto mulConstVal = mulConstVals[idx];

            newOutHighVals[idx] = outHighVal * mulConstVal;
            newOutLowVals[idx] = outLowVal * mulConstVal;
        }

        newOutLowContentAttr =
                Const::createFloatContentAttr(rewriter, fakeQuantOp.getLoc(), newOutRankedTensorType, newOutLowVals);
        newOutHighContentAttr =
                Const::createFloatContentAttr(rewriter, fakeQuantOp.getLoc(), newOutRankedTensorType, newOutHighVals);
    }

    auto newOutLowConst = rewriter.create<Const::DeclareOp>(outLowConst.getLoc(), newOutRankedTensorType,
                                                            std::move(newOutLowContentAttr));
    auto newOutHighConst = rewriter.create<Const::DeclareOp>(outHighConst.getLoc(), newOutRankedTensorType,
                                                             std::move(newOutHighContentAttr));

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(multiplyOp, fakeQuantOp.getInput(), fakeQuantOp.getInputLow(),
                                                    fakeQuantOp.getInputHigh(), newOutLowConst, newOutHighConst,
                                                    fakeQuantOp.getLevelsAttr(), fakeQuantOp.getLowFpTypeAttr(),
                                                    fakeQuantOp.getAutoBroadcastAttr());

    return mlir::success();
}

class FuseDQAndMul final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    FuseDQAndMul(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalToFuseDQ(IE::MultiplyOp multiplyOp) const;

private:
    Logger _log;
};

bool FuseDQAndMul::isLegalToFuseDQ(IE::MultiplyOp multiplyOp) const {
    bool lhsIsActivation = vpux::VPU::isEltwiseLhsActivation<IE::MultiplyOp>(multiplyOp);

    auto dequantizeOp = lhsIsActivation ? multiplyOp.getInput1().getDefiningOp<IE::DequantizeOp>()
                                        : multiplyOp.getInput2().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr || !dequantizeOp->hasOneUse()) {
        return false;
    }

    auto dequantInput = dequantizeOp.getInput();
    const auto isConstInputOp = mlir::isa_and_present<Const::DeclareOp>(dequantInput.getDefiningOp());
    const auto isBlockArg = mlir::isa_and_present<mlir::BlockArgument>(dequantInput);

    if (!isConstInputOp && !isBlockArg) {
        return false;
    }

    auto mulConstOp = lhsIsActivation ? multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>()
                                      : multiplyOp.getInput1().getDefiningOp<Const::DeclareOp>();
    if (mulConstOp == nullptr) {
        return false;
    }

    auto inputType = dequantInput.getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(inputType);
    if (!tensorType) {
        return false;
    }

    auto elemType = tensorType.getElementType();
    auto uniformQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType);

    if (!uniformQType && !perAxisQType) {
        return false;
    }

    const auto mulConstContent = mulConstOp.getContent();
    if (mulConstContent.isSplat()) {
        return true;
    }

    if (uniformQType) {
        return false;
    }

    auto mulConstShape = getShape(mulConstOp.getOutput());
    const auto mulNonOneAxisCount = std::count_if(mulConstShape.begin(), mulConstShape.end(), [](auto size) {
        return size > 1;
    });

    if (mulNonOneAxisCount != 1) {
        return false;
    }

    const auto firstNonOneValue = std::find_if(mulConstShape.begin(), mulConstShape.end(), [](auto value) {
        return value != 1;
    });

    const auto firstNonOneIndex =
            firstNonOneValue != mulConstShape.end() ? std::distance(mulConstShape.begin(), firstNonOneValue) : -1;

    if (perAxisQType) {
        const auto quantAxis = perAxisQType.getQuantizedDimension();
        const auto scalesSize = perAxisQType.getScales().size();
        return (quantAxis == firstNonOneIndex) &&
               (scalesSize == 1 || static_cast<size_t>(mulConstShape.totalSize()) == scalesSize);
    }
    return false;
}

/*
    Input is Constant Weights:

         Quantized Weights (Const)
         (scale = S, zp = Z)
                  |
                  v
         +----------------+                       Quantized Weights (Const)
         |   Dequantize   |                        (scale = S * C, zp = Z)
         +----------------+                                  |
                  |                                          v
                  v                  =====>          +----------------+
            +----------+                             |   Dequantize   |
            | Multiply | <--- C                      +----------------+
            +----------+                                     |
                  |                                          v
                  v



    Input is BlockArgument:


       Quantized Weights(BlockArgs)                   Quantized Weights
        (scale = S, zp = Z)                          (scale = S, zp = Z)
                |                                              |
                v                                              v
        +----------------+                           +------------------+
        |   Dequantize   |                           |  QuantizeCast    |
        +----------------+                           | (scale = S*C)    |
                |                                    +------------------+
                v                                              |
          +----------+              =====>                     v
          | Multiply | ← C                            +----------------+
          +----------+                                |   Dequantize   |
                |                                     +----------------+
                v

*/

mlir::LogicalResult FuseDQAndMul::matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const {
    if (!isLegalToFuseDQ(multiplyOp)) {
        return mlir::failure();
    }

    const auto lhsIsActivation = vpux::VPU::isEltwiseLhsActivation<IE::MultiplyOp>(multiplyOp);

    auto dequantOp = lhsIsActivation ? multiplyOp.getInput1().getDefiningOp<IE::DequantizeOp>()
                                     : multiplyOp.getInput2().getDefiningOp<IE::DequantizeOp>();
    auto mulConstOp = lhsIsActivation ? multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>()
                                      : multiplyOp.getInput1().getDefiningOp<Const::DeclareOp>();

    if (dequantOp == nullptr || mulConstOp == nullptr) {
        return mlir::failure();
    }

    auto dequantInput = dequantOp.getInput();
    auto inputType = dequantInput.getType();

    const auto mulConstVals = getConstContentVals(mulConstOp.getContent());

    const auto newInputType = [&]() -> mlir::Type {
        auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(inputType);
        if (!ndType) {
            return nullptr;
        }
        auto elemType = ndType.getElementType();

        if (mlir::isa<mlir::quant::UniformQuantizedType>(elemType)) {
            if (mulConstVals.size() != 1) {
                return nullptr;
            }
            return IE::rescaleUniformQuantizedType(inputType, mulConstVals.front());
        }

        if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
            auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType);
            auto newPerAxisQType = IE::rescaleUniformQuantizedPerAxisType(perAxisQType, mulConstVals);
            return ndType.changeElemType(newPerAxisQType);
        }

        return nullptr;
    }();

    if (newInputType == nullptr) {
        return mlir::failure();
    }

    auto newInputElemType = mlir::cast<vpux::NDTypeInterface>(newInputType).getElementType();

    auto dstElemTypeAttr = mlir::TypeAttr::get(newInputElemType);

    auto newDequantInput =
            rewriter.createOrFold<IE::QuantizeCastOp>(dequantOp.getLoc(), newInputType, dequantInput, dstElemTypeAttr);

    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(multiplyOp, multiplyOp.getType(), newDequantInput,
                                                  dequantOp.getDstElemTypeAttr());
    return mlir::success();
}

//
// FuseQuantizationMultiplyPass
//

class FuseQuantizationMultiplyPass final : public IE::impl::FuseQuantizationMultiplyBase<FuseQuantizationMultiplyPass> {
public:
    explicit FuseQuantizationMultiplyPass(const bool fuseFQAndMulWithNonConstInput, Logger log)
            : _fuseFQAndMulWithNonConstInput(fuseFQAndMulWithNonConstInput) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _fuseFQAndMulWithNonConstInput = false;
};

mlir::LogicalResult FuseQuantizationMultiplyPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (fuseFQAndMulWithNonConstInput.hasValue()) {
        _fuseFQAndMulWithNonConstInput = fuseFQAndMulWithNonConstInput.getValue();
    }

    return mlir::success();
}

void FuseQuantizationMultiplyPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseFQAndMul>(&ctx, _log, _fuseFQAndMulWithNonConstInput);
    patterns.add<FuseDQAndMul>(&ctx, _log);
    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseQuantizationMultiplyPass(const bool fuseFQAndMulWithNonConstInput,
                                                                         Logger log) {
    return std::make_unique<FuseQuantizationMultiplyPass>(fuseFQAndMulWithNonConstInput, log);
}
