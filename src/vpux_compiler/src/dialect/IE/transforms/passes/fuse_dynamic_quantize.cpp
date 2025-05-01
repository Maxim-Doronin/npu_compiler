//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEDYNAMICQUANTIZE
#define GEN_PASS_DEF_FUSEDYNAMICQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseDynamicQuantizePass
//

class FuseDynamicQuantizePass final : public IE::impl::FuseDynamicQuantizeBase<FuseDynamicQuantizePass> {
public:
    explicit FuseDynamicQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// The op sequence for the decomposed DynamicQuantizeLinear appears as follows when it reaches this point.
//
//  {
//    %0 = IE.ReduceMin(input)
//    %1 = IE.Clamp(%0)
//    %2 = IE.Subtract(%cst_1, %1)
//    %3 = IE.ReduceMax(input)
//    %4 = IE.Clamp(%3)
//    %5 = IE.Subtract(%4, %1)
//    %scale = IE.Multiply(%5, %cst_0)
//
//    %7 = IE.Divide(%2, %scale)
//    %8 = IE.Round(%7)
//    %9 = IE.Clamp(%8)
//    %zp = IE.Convert(%9)
//
//    %11 = IE.Multiply(input, %cst)
//    %12 = IE.Divide(%11, %5)
//    %13 = IE.Round(%12)
//    %14 = IE.Add(%13, %9)
//    %15 = IE.Clamp(%14)
//    %quantOutput = IE.Convert(%15)
//    return %quantOutput, %scale, %zp
//  }
//
// This sequence is similar to the quantization equation.
//    scale = (max-min) / 255
//    zp = (0-min) / scale
//    quantOutput = (x * 255) / (max-min) + zp
//                = x / scale + zp
//

void FuseDynamicQuantizePass::safeRunOnFunc() {
    auto func = getOperation();

    const auto isConstSplatEqual = [&](mlir::Value operand, double value) {
        auto constOp = operand.getDefiningOp<Const::DeclareOp>();
        auto convertOp = operand.getDefiningOp<IE::ConvertOp>();
        if (constOp == nullptr && convertOp == nullptr) {
            return false;
        }
        if (constOp == nullptr) {
            constOp = convertOp.getInput().getDefiningOp<Const::DeclareOp>();
        }
        if (constOp == nullptr) {
            return false;
        }
        auto splatValue = Const::getSplatValue<float>(constOp);
        return mlir::succeeded(splatValue) && isFloatEqual(splatValue.value(), value);
    };

    const auto createConvert = [&](mlir::OpBuilder& builder, mlir::Value newInput, mlir::Value origInput,
                                   mlir::Location loc) -> mlir::Value {
        const auto newEltType = newInput.getType().cast<NDTypeInterface>().getElementType();
        const auto origEltType = origInput.getType().cast<NDTypeInterface>().getElementType();
        if (newEltType == origEltType) {
            return newInput;
        }
        auto cvtOrigType = builder.create<IE::ConvertOp>(loc, newInput, origEltType);
        return cvtOrigType.getOutput();
    };

    for (auto reduceMinOp : llvm::make_early_inc_range(func.getOps<IE::ReduceMinOp>())) {
        const auto& nestLog = _log.nest();
        _log.trace("Got ReduceMinOp at {0}", reduceMinOp->getLoc());

        SmallVector<mlir::Operation*> opsToErase;
        if (!reduceMinOp->hasOneUse()) {
            nestLog.trace("ReduceMin has multi-users");
            return;
        }

        const auto reduceMinInput = reduceMinOp.getInput();
        const auto inputType = reduceMinInput.getType().getElementType();
        if (!inputType.isF32()) {
            nestLog.trace("ReduceMin has non-FP32 input");
            return;
        }

        // axes = [0,1,2,3..]
        auto reduceMinAxesValue = IE::extractAxes(reduceMinOp.getLoc(), reduceMinOp);
        if (reduceMinAxesValue.size() != getShape(reduceMinInput).size()) {
            nestLog.trace("ReduceMin doesn't calculate on all axes");
            return;
        }
        if (reduceMinOp.getKeepDims()) {
            nestLog.trace("ReduceMin keep_dims is true");
            return;
        }

        // 3 users: ReduceMax/ReduceMin/Multiply
        IE::ReduceMaxOp reduceMaxOp = nullptr;
        IE::MultiplyOp multiplyQuantSpanOp = nullptr;
        for (const auto userOp : reduceMinInput.getUsers()) {
            if (userOp == reduceMinOp.getOperation()) {
                continue;
            }
            if (mlir::isa<IE::ReduceMaxOp>(userOp) && reduceMaxOp == nullptr) {
                reduceMaxOp = mlir::cast<IE::ReduceMaxOp>(userOp);
                continue;
            }
            if (mlir::isa<IE::MultiplyOp>(userOp) && multiplyQuantSpanOp == nullptr) {
                multiplyQuantSpanOp = mlir::cast<IE::MultiplyOp>(userOp);
                continue;
            }

            nestLog.trace("ReduceMinOp/ReduceMaxOp/MultiplyOp don't have the same input");
            return;
        }

        if (reduceMaxOp == nullptr || multiplyQuantSpanOp == nullptr || !reduceMaxOp->hasOneUse() ||
            !multiplyQuantSpanOp->hasOneUse()) {
            nestLog.trace("ReduceMaxOp or MultiplyOp has multi-users");
            return;
        }

        // axes = [0,1,2,3..]
        auto reduceMaxAxesValue = IE::extractAxes(reduceMaxOp.getLoc(), reduceMaxOp);
        if (reduceMaxAxesValue.size() != getShape(reduceMaxOp.getInput()).size()) {
            nestLog.trace("ReduceMax doesn't calculate on all axes");
            return;
        }
        if (reduceMaxOp.getKeepDims()) {
            nestLog.trace("reduceMaxOp keep_dims is true");
            return;
        }

        opsToErase.push_back(multiplyQuantSpanOp);

        auto maxOp = mlir::dyn_cast<IE::ClampOp>(*reduceMaxOp->getUsers().begin());
        if (maxOp == nullptr || !maxOp->hasOneUse()) {
            nestLog.trace("ReduceMaxOp is not followed by ClampOp");
            return;
        }

        auto divideSpanOp = mlir::dyn_cast<IE::DivideOp>(*multiplyQuantSpanOp->getUsers().begin());
        if (divideSpanOp == nullptr || !divideSpanOp->hasOneUse()) {
            nestLog.trace("multiplyQuantSpanOp is not followed by DivideOp");
            return;
        }

        auto minOp = mlir::dyn_cast<IE::ClampOp>(*reduceMinOp->getUsers().begin());
        if (minOp == nullptr) {
            nestLog.trace("ReduceMinOp is not followed by ClampOp");
            return;
        }

        if (!isDoubleEqual(maxOp.getMinAttr().getValueAsDouble(), 0.0f) ||
            !isDoubleEqual(minOp.getMaxAttr().getValueAsDouble(), 0.0f)) {
            nestLog.trace("minOp or maxOp is not clamped in expected range");
            return;
        }

        opsToErase.append({maxOp, minOp});

        auto subtractSpanOp = mlir::dyn_cast<IE::SubtractOp>(*maxOp->getUsers().begin());
        if (subtractSpanOp == nullptr) {
            nestLog.trace("maxOp is not followed by SubtractOp");
            return;
        }

        // 2 users: Subtract/Subtract
        IE::SubtractOp subtractQuantSpanOp = nullptr;
        for (const auto userOp : minOp->getUsers()) {
            if (userOp == subtractSpanOp.getOperation()) {
                continue;
            }
            if (mlir::isa<IE::SubtractOp>(userOp) && subtractQuantSpanOp == nullptr) {
                subtractQuantSpanOp = mlir::cast<IE::SubtractOp>(userOp);
                continue;
            }
            return;
        }

        if (subtractQuantSpanOp == nullptr || !subtractQuantSpanOp->hasOneUse()) {
            nestLog.trace("One user of minOp is not SubtractOp");
            return;
        }
        if (!isConstSplatEqual(subtractQuantSpanOp.getInput1(), 0.0f)) {
            nestLog.trace("Input1 of subtractQuantSpanOp is not splat zero");
            return;
        }

        opsToErase.append({subtractQuantSpanOp, subtractSpanOp});

        // 2 users: Multiply/Divide
        IE::MultiplyOp multiplyScaleOp = nullptr;
        for (const auto userOp : subtractSpanOp->getUsers()) {
            if (userOp == divideSpanOp.getOperation()) {
                continue;
            }
            if (mlir::isa<IE::MultiplyOp>(userOp) && multiplyScaleOp == nullptr) {
                multiplyScaleOp = mlir::cast<IE::MultiplyOp>(userOp);
                continue;
            }
            return;
        }
        if (multiplyScaleOp == nullptr) {
            nestLog.trace("subtractSpanOp has no Multiply user");
            return;
        }

        opsToErase.append({multiplyScaleOp, divideSpanOp});

        auto divideQuant = mlir::dyn_cast<IE::DivideOp>(*subtractQuantSpanOp->getUsers().begin());
        if (divideQuant == nullptr || !divideQuant->hasOneUse()) {
            nestLog.trace("subtractQuantSpanOp is not followed by DivideOp");
            return;
        }

        if (divideQuant.getInput2().getDefiningOp() != multiplyScaleOp.getOperation()) {
            nestLog.trace("multiplyScaleOp is not one of inputs of divideQuant");
            return;
        }

        auto roundQuant = mlir::dyn_cast<IE::RoundOp>(*divideQuant->getUsers().begin());
        if (roundQuant == nullptr || !roundQuant->hasOneUse()) {
            nestLog.trace("divideQuant is not followed by RoundOp");
            return;
        }

        auto clampQuant = mlir::dyn_cast<IE::ClampOp>(*roundQuant->getUsers().begin());
        if (clampQuant == nullptr) {
            nestLog.trace("roundQuant is not followed by ClampOp");
            return;
        }

        opsToErase.append({divideQuant, roundQuant, clampQuant});

        // 2 users: Convert/Add
        IE::ConvertOp convertZp = nullptr;
        IE::AddOp addQuant = nullptr;
        for (const auto userOp : clampQuant->getUsers()) {
            if (mlir::isa<IE::AddOp>(userOp) && addQuant == nullptr) {
                addQuant = mlir::cast<IE::AddOp>(userOp);
                continue;
            }
            if (mlir::isa<IE::ConvertOp>(userOp) && convertZp == nullptr) {
                convertZp = mlir::cast<IE::ConvertOp>(userOp);
                continue;
            }
            return;
        }
        if (addQuant == nullptr || !addQuant->hasOneUse() || convertZp == nullptr) {
            nestLog.trace("clampQuant has no ConvertOp or AddOp user");
            return;
        }

        auto roundSpan = mlir::dyn_cast<IE::RoundOp>(*divideSpanOp->getUsers().begin());
        if (roundSpan == nullptr || !roundSpan->hasOneUse()) {
            nestLog.trace("divideSpanOp is not followed by RoundOp");
            return;
        }

        auto addZero = mlir::dyn_cast<IE::AddOp>(*roundSpan->getUsers().begin());
        if (addZero == nullptr || addZero != addQuant) {
            nestLog.trace("roundSpan is not followed by AddOp");
            return;
        }

        auto clampQuantZp = mlir::dyn_cast<IE::ClampOp>(*addZero->getUsers().begin());
        if (clampQuantZp == nullptr || !clampQuantZp->hasOneUse()) {
            nestLog.trace("addZero is not followed by ClampOp");
            return;
        }

        auto outputConvert = mlir::dyn_cast<IE::ConvertOp>(*clampQuantZp->getUsers().begin());
        if (outputConvert == nullptr) {
            nestLog.trace("clampQuantZp is not followed by ConvertOp");
            return;
        }

        opsToErase.append({roundSpan, addZero, clampQuantZp, outputConvert, convertZp});

        if (!isConstSplatEqual(multiplyQuantSpanOp.getInput2(), 255.0f)) {
            nestLog.info("Input2 of multiplyQuantSpanOp is not splat");
            return;
        }

        if (!isConstSplatEqual(multiplyScaleOp.getInput2(), static_cast<float>(1.0f / 255.0f))) {
            nestLog.info("Input2 of multiplyScaleOp is not splat");
            return;
        }

        // Scale
        auto outputScale = multiplyScaleOp.getOutput();
        // Zero point
        auto outputZp = convertZp.getOutput();
        // data
        auto outputQuant = outputConvert.getOutput();

        if (getShape(outputScale).size() != 1 || getShape(outputZp).size() != 1) {
            nestLog.trace("Rank of Scale or ZP is not one");
            return;
        }

        auto builder = mlir::OpBuilder(reduceMinOp);
        builder.setInsertionPointAfter(reduceMinOp);
        if (reduceMinOp->isBeforeInBlock(reduceMaxOp)) {
            builder.setInsertionPointAfter(reduceMaxOp);
        }

        const auto loc = outputConvert->getLoc();
        auto dqOp = builder.create<IE::DynamicQuantizeOp>(appendLoc(loc, "_dq_linear"), reduceMinInput,
                                                          reduceMinOp.getOutput(), reduceMaxOp.getOutput());
        auto dqZp = createConvert(builder, dqOp.getZeroPoint(), outputZp, appendLoc(loc, "_zp"));
        auto dqOutput = createConvert(builder, dqOp.getOutput(), outputQuant, appendLoc(loc, "_output"));
        auto dqScale = createConvert(builder, dqOp.getScale(), outputScale, appendLoc(loc, "_scale"));

        outputQuant.replaceAllUsesWith(dqOutput);
        outputZp.replaceAllUsesWith(dqZp);
        outputScale.replaceAllUsesWith(dqScale);

        _log.trace("DynamicQuantizeOp fused");

        for (auto op : opsToErase | reversed) {
            op->erase();
        }
    }
}

}  // namespace

//
// createFuseDynamicQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseDynamicQuantizePass(Logger log) {
    return std::make_unique<FuseDynamicQuantizePass>(log);
}
