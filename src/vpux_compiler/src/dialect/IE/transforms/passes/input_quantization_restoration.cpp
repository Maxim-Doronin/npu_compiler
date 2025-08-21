//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_INPUTQUANTIZATIONRESTORATION
#define GEN_PASS_DEF_INPUTQUANTIZATIONRESTORATION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int levels = QuantizationLevels::QUANT_LEVELS_8BIT;

//
// InputQuantizationRestoration
//

class InputQuantizationRestoration final :
        public IE::impl::InputQuantizationRestorationBase<InputQuantizationRestoration> {
public:
    explicit InputQuantizationRestoration(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// InputQuantizationRestoration
//

using FQParametersType = std::tuple<mlir::Value,  // input low
                                    mlir::Value,  // input high
                                    mlir::Value,  // output low
                                    mlir::Value   // output high
                                    >;

FQParametersType createQuantizationConstants(mlir::PatternRewriter& rewriter, float scale, float zeroPoint,
                                             mlir::Location loc, mlir::ArrayRef<int64_t> shape) {
    // Calculate low and high values using the formulas
    float inputLowValue = -scale * zeroPoint;
    float inputHighValue = scale * ((levels - 1) - zeroPoint);
    float outputLowValue = inputLowValue;
    float outputHighValue = inputHighValue;

    // Create a shape vector of the same rank, all ones
    std::vector<int64_t> onesShape(shape.size(), 1);

    // Use this shape for the constant tensor type
    auto tensorType = mlir::RankedTensorType::get(onesShape, rewriter.getF32Type());

    auto inputLow = Const::createConst(rewriter, loc, tensorType, ArrayRef(inputLowValue));
    auto inputHigh = Const::createConst(rewriter, loc, tensorType, ArrayRef(inputHighValue));
    auto outputLow = Const::createConst(rewriter, loc, tensorType, ArrayRef(outputLowValue));
    auto outputHigh = Const::createConst(rewriter, loc, tensorType, ArrayRef(outputHighValue));

    return std::make_tuple(inputLow, inputHigh, outputLow, outputHigh);
}

class ConvertMultiplyPattern final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    ConvertMultiplyPattern(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const final {
        _log.debug("Checking MultiplyOp at {0}", multiplyOp->getLoc());

        // Check if the first or second input is a ConvertOp
        IE::ConvertOp convertOp = nullptr;
        mlir::Value otherOperand = nullptr;
        auto firstMultiplyInput = multiplyOp.getOperand(0).getDefiningOp<IE::ConvertOp>();
        auto secondMultiplyInput = multiplyOp.getOperand(1).getDefiningOp<IE::ConvertOp>();

        if (firstMultiplyInput != nullptr) {
            convertOp = firstMultiplyInput;
            otherOperand = multiplyOp.getOperand(1);
        } else if (secondMultiplyInput != nullptr) {
            convertOp = secondMultiplyInput;
            otherOperand = multiplyOp.getOperand(0);
        } else {
            return mlir::failure();
        }

        // Get the type and shape of the other operand
        auto otherType = mlir::dyn_cast<mlir::RankedTensorType>(otherOperand.getType());
        if (!otherType) {
            return mlir::failure();
        }
        auto otherShape = otherType.getShape();

        // Check that all dimensions are 1 - required for FakeQuantize operation
        for (size_t i = 1; i < otherShape.size(); ++i) {
            if (otherShape[i] != 1) {
                return mlir::failure();
            }
        }

        // Check that convert input is U8
        if (mlir::cast<vpux::NDTypeInterface>(convertOp.getInput().getType()).getElementType() !=
            vpux::getUInt8Type(rewriter.getContext())) {
            return mlir::failure();
        }

        // Check if Convert input is the function/network input
        auto convertInput = convertOp.getInput();
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(convertInput);
        if (blockArg == nullptr) {
            return mlir::failure();
        }

        // Check if any user of Multiply is a FakeQuantize
        for (auto* user : multiplyOp->getUsers()) {
            if (mlir::isa<IE::FakeQuantizeOp>(user)) {
                return mlir::failure();
            }
        }

        // Set the insertion point just after ConvertOp
        rewriter.setInsertionPointAfter(convertOp);
        // Get scale and zero point
        auto scaleConstOp = multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>();
        if (scaleConstOp == nullptr) {
            return mlir::failure();
        }
        auto scaleValueOr = vpux::Const::getSplatValue<float>(scaleConstOp);
        if (mlir::failed(scaleValueOr)) {
            return mlir::failure();
        }
        auto scaleValue = scaleValueOr.value();

        // Use 0 for zero point as we do not process subgraph with Subtract
        const float zeroPoint = 0.0f;

        // Get the input shape
        auto inputShape = mlir::cast<mlir::RankedTensorType>(multiplyOp.getInput1().getType()).getShape();

        auto [inputLow, inputHigh, outputLow, outputHigh] =
                createQuantizationConstants(rewriter, scaleValue, zeroPoint, multiplyOp->getLoc(), inputShape);

        rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
                multiplyOp, multiplyOp.getInput1(), inputLow, inputHigh, outputLow, outputHigh,
                rewriter.getI64IntegerAttr(levels), nullptr,
                vpux::IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), vpux::IE::AutoBroadcastType::NUMPY));

        return mlir::success();
    }

private:
    Logger _log;
};

//
// safeRunOnFunc
//

void InputQuantizationRestoration::safeRunOnFunc() {
    auto func = getOperation();

    mlir::RewritePatternSet patterns(func.getContext());
    patterns.add<ConvertMultiplyPattern>(func.getContext(), _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createInputQuantizationRestoration
//

std::unique_ptr<mlir::Pass> vpux::IE::createInputQuantizationRestorationPass(Logger log) {
    return std::make_unique<InputQuantizationRestoration>(log);
}
