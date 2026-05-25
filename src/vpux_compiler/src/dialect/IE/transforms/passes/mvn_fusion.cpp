//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MVNFUSION
#define GEN_PASS_DEF_MVNFUSION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr double EPS_THRESHOLD = 1e-4;

//
// Helpers
//

std::optional<Shape> canConvertToMVN1(ShapeRef inputShapeRef, ArrayRef<int64_t> axes, bool& isAcrossChannel) {
    const auto inputRank = inputShapeRef.size();
    const auto inputShape = inputShapeRef.raw();

    isAcrossChannel = false;
    if (inputRank == 2 && axes.size() == 1 && axes[0] == 1) {
        // HxW -> 1xHxWx1
        return Shape{1, inputShape[0], inputShape[1], 1};
    } else if (inputRank == 3) {
        if (axes.size() == 1 && axes[0] == 2) {
            // CxHxW -> CxHxWx1
            return Shape{inputShape[0], inputShape[1], inputShape[2], 1};
        } else if (axes.size() == 1 && axes[0] == 1 && inputShape[2] == 1) {
            // AxBx1 with axis [1] -> Ax1xBx1, MVN1 (across_channels=false) normalizes over H,W = B,1
            return Shape{inputShape[0], 1, inputShape[1], inputShape[2]};
        } else if (axes.size() == 2 && axes[0] == 1 && axes[1] == 2) {
            // CxHxW -> 1xCxHxW
            return Shape{1, inputShape[0], inputShape[1], inputShape[2]};
        } else if (axes.size() == 2 && axes[0] == 0 && axes[1] == 2 && inputShape[0] == 1) {
            // 1xHxW with axes [0,2]: dim-0 is trivially-sized, equivalent to axes [2]
            // 1xHxW -> 1xHx1xW
            return Shape{inputShape[0], inputShape[1], 1, inputShape[2]};
        }
    } else if (inputRank == 4) {
        if (axes.size() == 3 && axes[0] == 1 && axes[1] == 2 && axes[2] == 3) {
            isAcrossChannel = true;
            return inputShape;
        } else if (axes.size() == 2 && axes[0] == 2 && axes[1] == 3) {
            return inputShape;
        }
    }
    return std::nullopt;
}

std::optional<double> getEpsValue(mlir::Value epsInput, bool isOutsideEps) {
    auto convertOp = epsInput.getDefiningOp<IE::ConvertOp>();
    auto constOp = epsInput.getDefiningOp<Const::DeclareOp>();
    if (convertOp) {
        constOp = convertOp.getInput().getDefiningOp<Const::DeclareOp>();
    }
    if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
        return std::nullopt;
    }
    const auto epsContent = constOp.getContent();
    const auto epsValue = epsContent.getSplatValue<double>();
    if (isOutsideEps && epsValue > EPS_THRESHOLD) {
        return std::nullopt;
    }
    return epsValue;
}

void normalizeAndSortAxes(SmallVector<int64_t>& axes, int64_t rank) {
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] < 0) {
            axes[i] += rank;
        }
    }
    std::sort(axes.begin(), axes.end());
}

mlir::Value skipShapeOpsBackward(mlir::Value val) {
    while (auto defOp = val.getDefiningOp()) {
        if (!mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp, IE::TransposeOp>(defOp) || !defOp->hasOneUse()) {
            break;
        }
        val = defOp->getOperand(0);
    }
    return val;
}

mlir::Value skipShapeOpsForward(mlir::Value val) {
    while (val.hasOneUse()) {
        auto user = *val.getUsers().begin();
        if (!mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp, IE::TransposeOp>(user)) {
            break;
        }
        val = user->getResult(0);
    }
    return val;
}

// Returns true when val is a scalar splat constant with value 0.
// Handles optional ConvertOp wrapping a Const::DeclareOp.
bool isZeroSplatConst(mlir::Value val) {
    auto convertOp = val.getDefiningOp<IE::ConvertOp>();
    auto constOp = val.getDefiningOp<Const::DeclareOp>();
    if (convertOp != nullptr) {
        constOp = convertOp.getInput().getDefiningOp<Const::DeclareOp>();
    }
    if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
        return false;
    }

    return isDoubleEqual(constOp.getContent().getSplatValue<double>(), 0.0);
}

//
// MVNFusion
//

class MVNFusion final : public mlir::OpRewritePattern<IE::DivideOp> {
public:
    MVNFusion(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DivideOp>(ctx), _log(log) {
        setDebugName("MVNFusion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// This pass convert this subgraph
//    (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean(x^2, axes) - (ReduceMean(x, axes) ^ 2)) + eps)
// or
//    (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean(x^2, axes) - (ReduceMean(x, axes) ^ 2) + eps))
// to a single MVN1
//

mlir::LogicalResult MVNFusion::matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Divide '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto meanSubOp = origOp.getInput1().getDefiningOp<IE::SubtractOp>();
    if (meanSubOp == nullptr) {
        return matchFailed(rewriter, origOp, "No x SubtractOp found");
    }

    auto inputMeanOp = meanSubOp.getInput2().getDefiningOp<IE::ReduceMeanOp>();
    if (inputMeanOp == nullptr) {
        return matchFailed(rewriter, origOp, "No x ReduceMeanOp found");
    }
    auto inputMeanAxesValue = IE::extractAxes(origOp.getLoc(), inputMeanOp);

    if (inputMeanOp.getInput() != meanSubOp.getInput1()) {
        return matchFailed(rewriter, origOp, "Not the same input");
    }
    const auto mvnInput = inputMeanOp.getInput();
    const auto inputShape = getShape(mvnInput);
    const auto inputRank = inputShape.size();
    if (inputRank < 2 || inputRank > 4) {
        return matchFailed(rewriter, origOp, "Invalid input shape rank");
    }

    // inside-sqrt or outside-sqrt
    auto insideEpsSqrtOp = origOp.getInput2().getDefiningOp<IE::SqrtOp>();
    auto outsideEpsAddOp = origOp.getInput2().getDefiningOp<IE::AddOp>();

    IE::SubtractOp squareSubOp = nullptr;
    mlir::Value epsInput;
    bool isOutsideEps;
    if (insideEpsSqrtOp) {
        auto insideEpsAddOp = insideEpsSqrtOp.getInput().getDefiningOp<IE::AddOp>();
        if (insideEpsAddOp == nullptr) {
            return matchFailed(rewriter, origOp, "No inside-eps AddOp found");
        }
        squareSubOp = insideEpsAddOp.getInput1().getDefiningOp<IE::SubtractOp>();
        if (squareSubOp == nullptr) {
            return matchFailed(rewriter, origOp, "No inside-eps SubtractOp found");
        }

        epsInput = insideEpsAddOp.getInput2();
        isOutsideEps = false;
    } else if (outsideEpsAddOp) {
        auto outsideEpsSqrtOp = outsideEpsAddOp.getInput1().getDefiningOp<IE::SqrtOp>();
        if (outsideEpsSqrtOp == nullptr) {
            return matchFailed(rewriter, origOp, "No outside-eps SqrtOp found");
        }
        squareSubOp = outsideEpsSqrtOp.getInput().getDefiningOp<IE::SubtractOp>();
        if (squareSubOp == nullptr) {
            return matchFailed(rewriter, origOp, "No outside-eps SubtractOp found");
        }

        epsInput = outsideEpsAddOp.getInput2();
        isOutsideEps = true;
    } else {
        return matchFailed(rewriter, origOp, "No inside-eps or outside-eps mode found");
    }

    if (squareSubOp == nullptr) {
        return matchFailed(rewriter, origOp, "No square SubtractOp found");
    }

    auto epsValueOpt = getEpsValue(epsInput, isOutsideEps);
    if (!epsValueOpt.has_value()) {
        return matchFailed(rewriter, origOp, "No valid eps found");
    }
    const auto epsValue = epsValueOpt.value();

    auto isMultiplySquare = [](IE::MultiplyOp op) {
        return op.getInput1() == op.getInput2();
    };

    auto squareMeanOp = squareSubOp.getInput1().getDefiningOp<IE::ReduceMeanOp>();
    if (squareMeanOp == nullptr) {
        return matchFailed(rewriter, origOp, "No square ReduceMeanOp found");
    }
    auto squareMeanAxesValue = IE::extractAxes(origOp.getLoc(), squareMeanOp);

    if (inputMeanAxesValue != squareMeanAxesValue) {
        return matchFailed(rewriter, origOp, "ReduceMean ops have different axes");
    }

    auto squareOp = squareMeanOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (squareOp == nullptr) {
        return matchFailed(rewriter, origOp, "No x MultiplyOp found");
    }
    if (!isMultiplySquare(squareOp)) {
        return matchFailed(rewriter, origOp, "x MultiplyOp is not square");
    }
    if (squareOp.getInput1() != mvnInput) {
        return matchFailed(rewriter, origOp, "Not the same input");
    }

    auto meanSquareOp = squareSubOp.getInput2().getDefiningOp<IE::MultiplyOp>();
    if (meanSquareOp == nullptr) {
        return matchFailed(rewriter, origOp, "No MultiplyOp for ReduceMean found");
    }
    if (!isMultiplySquare(meanSquareOp)) {
        return matchFailed(rewriter, origOp, "MultiplyOp for ReduceMean is not square");
    }
    auto meanToSquareOp = meanSquareOp.getInput1().getDefiningOp<IE::ReduceMeanOp>();
    if (meanToSquareOp == nullptr) {
        return matchFailed(rewriter, origOp, "No ReduceMeanOp for square found");
    }
    if (meanToSquareOp != inputMeanOp) {
        return matchFailed(rewriter, origOp, "Not the same ReduceMean input");
    }

    normalizeAndSortAxes(inputMeanAxesValue, inputRank);

    bool isAcrossChannel;
    const auto newShapeOpt = canConvertToMVN1(inputShape, inputMeanAxesValue, isAcrossChannel);
    if (!newShapeOpt.has_value()) {
        return matchFailed(rewriter, origOp, "Cannot convert to mvn");
    }

    const auto ctx = origOp.getContext();
    auto preReshapeOp = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "in_reshape"), inputMeanOp.getInput(),
                                                       getIntArrayAttr(ctx, newShapeOpt.value()));

    const auto normVarianceAttr = mlir::BoolAttr::get(ctx, true);
    const auto acrossChannelsAttr = mlir::BoolAttr::get(ctx, isAcrossChannel);
    const auto epsAttr = getFPAttr(ctx, epsValue);

    auto mvnOp = rewriter.create<IE::MVNOp>(origOp.getLoc(), preReshapeOp.getOutput(), acrossChannelsAttr,
                                            normVarianceAttr, epsAttr);

    _log.trace("Replace '{0}' with new op '{1}'", origOp.getLoc(), mvnOp);
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, mvnOp.getOutput(),
                                                                 getIntArrayAttr(ctx, getShape(origOp.getOutput())));
    extendOpLoc(outReshape, "out_reshape");
    return mlir::success();
}

//
// MVNFusionOvRef
//

class MVNFusionOvRef final : public mlir::OpRewritePattern<IE::DivideOp> {
public:
    MVNFusionOvRef(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DivideOp>(ctx), _log(log) {
        setDebugName("MVNFusionOvRef");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// Converts the typical OV MVN decomposed-subgraph back to MVN1
//   D = x - ReduceMean(x, axes)
// out = D /  Sqrt(ReduceMean(D ^ 2) + eps), or
//     = D / (Sqrt(ReduceMean(D ^ 2)      ) + eps)
//

mlir::LogicalResult MVNFusionOvRef::matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Divide '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto meanSubOp = origOp.getInput1().getDefiningOp<IE::SubtractOp>();
    if (meanSubOp == nullptr) {
        return matchFailed(rewriter, origOp, "No SubtractOp found");
    }

    auto inputMeanOp = meanSubOp.getInput2().getDefiningOp<IE::ReduceMeanOp>();
    if (inputMeanOp == nullptr) {
        return matchFailed(rewriter, origOp, "No ReduceMeanOp(x) found");
    }
    auto inputMeanAxesValue = IE::extractAxes(origOp.getLoc(), inputMeanOp);

    if (inputMeanOp.getInput() != meanSubOp.getInput1()) {
        return matchFailed(rewriter, origOp, "No (x - ReduceMeanOp(x)) found");
    }
    const auto mvnInput = inputMeanOp.getInput();
    const auto inputShape = getShape(mvnInput);
    const auto inputRank = inputShape.size();
    if (inputRank < 2 || inputRank > 4) {
        return matchFailed(rewriter, origOp, "Invalid input shape rank");
    }

    // inside-sqrt or outside-sqrt
    auto insideEpsSqrtOp = origOp.getInput2().getDefiningOp<IE::SqrtOp>();
    auto outsideEpsAddOp = origOp.getInput2().getDefiningOp<IE::AddOp>();

    mlir::Value epsInput;
    mlir::Operation* reduceMean2 = nullptr;
    bool isOutsideEps;
    if (insideEpsSqrtOp) {
        auto insideEpsAddOp = insideEpsSqrtOp.getInput().getDefiningOp<IE::AddOp>();
        if (insideEpsAddOp == nullptr) {
            return matchFailed(rewriter, origOp, "No inside-eps AddOp found");
        }
        epsInput = insideEpsAddOp.getInput2();
        reduceMean2 = insideEpsAddOp.getInput1().getDefiningOp();
        isOutsideEps = false;
    } else if (outsideEpsAddOp) {
        auto outsideEpsSqrtOp = outsideEpsAddOp.getInput1().getDefiningOp<IE::SqrtOp>();
        if (outsideEpsSqrtOp == nullptr) {
            return matchFailed(rewriter, origOp, "No outside-eps SqrtOp found");
        }
        epsInput = outsideEpsAddOp.getInput2();
        reduceMean2 = outsideEpsSqrtOp.getInput().getDefiningOp();
        isOutsideEps = true;
    } else {
        return matchFailed(rewriter, origOp, "No inside-eps or outside-eps mode found");
    }

    auto epsValueOpt = getEpsValue(epsInput, isOutsideEps);
    if (!epsValueOpt.has_value()) {
        return matchFailed(rewriter, origOp, "No valid eps found");
    }
    const auto epsValue = epsValueOpt.value();

    VPUX_THROW_UNLESS(reduceMean2, "Checked Add/Sqrt op (mandatory inputs) has nullptr input");

    auto squareMeanOp = mlir::dyn_cast<IE::ReduceMeanOp>(*reduceMean2);
    if (squareMeanOp == nullptr) {
        return matchFailed(rewriter, origOp, "No square ReduceMeanOp found");
    }
    auto squareMeanAxesValue = IE::extractAxes(origOp.getLoc(), squareMeanOp);

    if (inputMeanAxesValue != squareMeanAxesValue) {
        return matchFailed(rewriter, origOp, "ReduceMean ops have different axes");
    }

    // detect Pow(x,2) or Multiply(x,x)
    auto getSquareOp = [](mlir::Operation* op) -> mlir::Operation* {
        if (auto power = mlir::dyn_cast<IE::PowerOp>(*op)) {
            auto constOp = power.getInput2().getDefiningOp<Const::DeclareOp>();
            if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
                return nullptr;
            }
            const auto coefContent = constOp.getContent();
            const auto coefValue = coefContent.getSplatValue<double>();
            return (coefValue == 2.0) ? op : nullptr;
        } else if (auto mul = mlir::dyn_cast<IE::MultiplyOp>(op)) {
            return mul.getInput1() == mul.getInput2() ? op : nullptr;
        }
        return nullptr;
    };

    mlir::Operation* squareOp = getSquareOp(squareMeanOp.getInput().getDefiningOp());
    if (squareOp == nullptr) {
        return matchFailed(rewriter, origOp, "No Squaring op found");
    }
    auto squareSubOp = mlir::dyn_cast<IE::SubtractOp>(squareOp->getOperand(0).getDefiningOp());
    if (squareSubOp == nullptr) {
        return matchFailed(rewriter, origOp, "Squaring op input is not a SubtractOp");
    }
    if (squareSubOp != meanSubOp) {
        // Allow semantically equivalent SubtractOps:
        // both must compute (x - ReduceMean(x, same_axes)) from the same input
        auto squareSubMeanOp = squareSubOp.getInput2().getDefiningOp<IE::ReduceMeanOp>();
        if (squareSubMeanOp == nullptr || squareSubOp.getInput1() != meanSubOp.getInput1() ||
            squareSubMeanOp.getInput() != inputMeanOp.getInput() ||
            IE::extractAxes(origOp.getLoc(), squareSubMeanOp) != inputMeanAxesValue) {
            return matchFailed(rewriter, origOp, "Subtract->SquareOp link not found");
        }
    }

    normalizeAndSortAxes(inputMeanAxesValue, inputRank);

    bool isAcrossChannel;
    const auto newShapeOpt = canConvertToMVN1(inputShape, inputMeanAxesValue, isAcrossChannel);
    if (!newShapeOpt.has_value()) {
        return matchFailed(rewriter, origOp, "Cannot convert to mvn");
    }

    const auto ctx = origOp.getContext();
    auto preReshapeOp = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "in_reshape"), inputMeanOp.getInput(),
                                                       getIntArrayAttr(ctx, newShapeOpt.value()));

    const auto normVarianceAttr = mlir::BoolAttr::get(ctx, true);
    const auto acrossChannelsAttr = mlir::BoolAttr::get(ctx, isAcrossChannel);
    const auto epsAttr = getFPAttr(ctx, epsValue);

    auto mvnOp = rewriter.create<IE::MVNOp>(origOp.getLoc(), preReshapeOp.getOutput(), acrossChannelsAttr,
                                            normVarianceAttr, epsAttr);

    _log.trace("Replace '{0}' with new op '{1}'", origOp.getLoc(), mvnOp);
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, mvnOp.getOutput(),
                                                                 getIntArrayAttr(ctx, getShape(origOp.getOutput())));
    extendOpLoc(outReshape, "out_reshape");
    return mlir::success();
}

//
// MVN1Mapping
//
// Describes how to reshape (and optionally transpose) an input tensor so that
// IE::MVNOp can be applied, then undo the transformation on the output.
// When preTransposePerm is empty no Transpose op is emitted.
//

struct MVN1Mapping {
    Shape mvnShape;                           // 4D shape fed into IE::MVNOp
    bool isAcrossChannel;                     // IE::MVNOp across-channels flag
    SmallVector<uint32_t> preTransposePerm;   // empty = no pre-transpose
    Shape postReshapeShape;                   // 3D shape after MVN + reshape, before post-transpose
    SmallVector<uint32_t> postTransposePerm;  // empty = no post-transpose
};

// Returns an MVN1Mapping for the given (inputShape, normalizedAxes) pair.
// Falls back to a Transpose-based path when a pure Reshape is insufficient.
static std::optional<MVN1Mapping> getMVN1Mapping(ShapeRef inputShape, ArrayRef<int64_t> axes) {
    // --- Direct (no-transpose) path ---
    bool isAcrossChannel = false;
    const auto directShape = canConvertToMVN1(inputShape, axes, isAcrossChannel);
    if (directShape.has_value()) {
        return MVN1Mapping{directShape.value(), isAcrossChannel, {}, Shape{}, {}};
    }

    const auto rank = inputShape.size();
    const auto s = inputShape.raw();

    // --- Transpose path: rank-3 shape CxHxW, axes [0,1] ---
    // A pure reshape cannot bring the C values into contiguous MVN spatial positions, so a Transpose is required:
    //   Pre:  Transpose [2,0,1]  : CxHxW → WxCxH
    //         Reshape            : WxCxH → 1xWxCxH
    //   MVN(across_channels=false): normalizes over CxH per channel → correct
    //   Post: Reshape            : 1xWxCxH → WxCxH
    //         Transpose [1,2,0]  : WxCxH → CxHxW  (self-inverse)
    if (rank == 3 && axes.size() == 2 && axes[0] == 0 && axes[1] == 1) {
        return MVN1Mapping{
                Shape{1, s[2], s[0], s[1]},
                /*isAcrossChannel=*/false,  SmallVector<uint32_t>{2, 0, 1},
                Shape{s[2], s[0], s[1]},    SmallVector<uint32_t>{1, 2, 0},
        };
    }

    return std::nullopt;
}

//
// MVNFusionWithSquaredDiff
//

class MVNFusionWithSquaredDiff final : public mlir::OpRewritePattern<IE::SquaredDifferenceOp> {
public:
    MVNFusionWithSquaredDiff(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SquaredDifferenceOp>(ctx), _log(log) {
        setDebugName("MVNFusionWithSquaredDiff");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SquaredDifferenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// Converts the TFLite-style LayerNorm decomposition to IE::MVNOp.
// Matches the subgraph:
//
//   μ        = ReduceMean(x, axes)
//   σ²       = ReduceMean(SquaredDifference(x, Reshape*(μ)), axes)
//   r        = Power(σ² + ε, -0.5)
//   out      = Multiply(x, Reshape*(r)) + Reshape(Subtract(0, Multiply(μ, r)))
//            = (x − μ) / sqrt(σ² + ε)            [= MVN]
//
// Anchor: IE::SquaredDifferenceOp
//

mlir::LogicalResult MVNFusionWithSquaredDiff::matchAndRewrite(IE::SquaredDifferenceOp sqDiffOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got SquaredDifferenceOp '{0}' at '{1}'", sqDiffOp->getName(), sqDiffOp->getLoc());

    // --- 1. port1 of SquaredDiff = ShapeOps*(ReduceMean(x, axes)) ---
    auto inputMeanOp = skipShapeOpsBackward(sqDiffOp.getInput2()).getDefiningOp<IE::ReduceMeanOp>();
    if (inputMeanOp == nullptr) {
        return matchFailed(rewriter, sqDiffOp, "port1 does not trace back to ReduceMeanOp");
    }

    // --- 2. port0 of SquaredDiff = x (same source as ReduceMean input) ---
    const mlir::Value xVal = sqDiffOp.getInput1();
    if (skipShapeOpsBackward(inputMeanOp.getInput()) != skipShapeOpsBackward(xVal)) {
        return matchFailed(rewriter, sqDiffOp, "SquaredDifference inputs do not share the same source");
    }
    const mlir::Value mvnInput = inputMeanOp.getInput();
    const auto inputShape = getShape(mvnInput);
    const auto inputRank = inputShape.size();
    if (inputRank < 2 || inputRank > 4) {
        return matchFailed(rewriter, sqDiffOp, "Input rank is outside supported range [2, 4]");
    }

    // --- 3. SquaredDifference must have exactly one consumer ---
    if (!sqDiffOp->hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "SquaredDifferenceOp has multiple uses");
    }

    // --- 4. Single consumer of SquaredDiff: ReduceMean(sq_diff, axes) = variance ---
    auto varMeanOp = mlir::dyn_cast<IE::ReduceMeanOp>(*sqDiffOp->getUsers().begin());
    if (varMeanOp == nullptr || !varMeanOp->hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "Expected single-use ReduceMeanOp after SquaredDifference");
    }
    auto inputMeanAxes = IE::extractAxes(sqDiffOp.getLoc(), inputMeanOp);
    auto varMeanAxes = IE::extractAxes(sqDiffOp.getLoc(), varMeanOp);
    if (inputMeanAxes != varMeanAxes) {
        return matchFailed(rewriter, sqDiffOp, "Mean and variance ReduceMean ops have different axes");
    }

    // --- 5. Add(variance, eps) ---
    auto addEpsOp = mlir::dyn_cast<IE::AddOp>(*varMeanOp->getUsers().begin());
    if (addEpsOp == nullptr || !addEpsOp->hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "Expected single-use AddOp after variance ReduceMean");
    }
    const mlir::Value epsInput =
            addEpsOp.getInput1().getDefiningOp() == varMeanOp ? addEpsOp.getInput2() : addEpsOp.getInput1();
    const auto epsOpt = getEpsValue(epsInput, /*isOutsideEps=*/false);
    if (!epsOpt.has_value()) {
        return matchFailed(rewriter, sqDiffOp, "Cannot extract epsilon value");
    }

    // --- 6. Power(var + eps, -0.5) = rsqrt ---
    auto rsqrtOp = mlir::dyn_cast<IE::PowerOp>(*addEpsOp->getUsers().begin());
    if (rsqrtOp == nullptr) {
        return matchFailed(rewriter, sqDiffOp, "Expected PowerOp after Add(var, eps)");
    }

    // Accept both Const and ConvertOp(Const) as the exponent source
    auto expConst = rsqrtOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (expConst == nullptr) {
        if (auto cvt = rsqrtOp.getInput2().getDefiningOp<IE::ConvertOp>()) {
            expConst = cvt.getInput().getDefiningOp<Const::DeclareOp>();
        }
    }
    if (expConst == nullptr || !expConst.getContentAttr().isSplat()) {
        return matchFailed(rewriter, sqDiffOp, "Power exponent is not a splat constant");
    }
    const auto expVal = expConst.getContent().getSplatValue<double>();
    if (!isDoubleEqual(expVal, -0.5)) {
        return matchFailed(rewriter, sqDiffOp, "Power exponent is not -0.5");
    }

    // --- 7. rsqrtOp must have exactly two users:
    //        - a shape-chain head (Reshape/AffineReshape) leading to Multiply(x, rsqrt_r)
    //        - Multiply(mean, rsqrt) leading to the negated-mean branch
    //    An optional Transpose may follow the Reshape before reaching each Multiply.
    mlir::Operation* reshapeRsqrtHeadOp = nullptr;
    IE::MultiplyOp mulMeanRsqrtOp = nullptr;
    for (auto* user : rsqrtOp->getUsers()) {
        if (mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp>(user)) {
            if (reshapeRsqrtHeadOp != nullptr) {
                return matchFailed(rewriter, sqDiffOp, "Multiple Reshape users of PowerOp");
            }
            reshapeRsqrtHeadOp = user;
        } else if (auto mulOp = mlir::dyn_cast<IE::MultiplyOp>(user)) {
            if (mulMeanRsqrtOp != nullptr) {
                return matchFailed(rewriter, sqDiffOp, "Multiple Multiply users of PowerOp");
            }
            mulMeanRsqrtOp = mulOp;
        } else {
            return matchFailed(rewriter, sqDiffOp, "Unexpected user type of PowerOp");
        }
    }
    if (reshapeRsqrtHeadOp == nullptr || mulMeanRsqrtOp == nullptr) {
        return matchFailed(rewriter, sqDiffOp,
                           "Expected one Reshape/AffineReshape and one Multiply as PowerOp(-0.5) users");
    }
    if (!reshapeRsqrtHeadOp->hasOneUse() || !mulMeanRsqrtOp->hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "Reshape or Multiply after PowerOp has multiple uses");
    }

    // --- 8. Verify Multiply(mean, rsqrt): inputs must be (inputMeanOp result, rsqrtOp result) ---
    const mlir::Value meanResult = inputMeanOp.getOutput();
    const mlir::Value rsqrtResult = rsqrtOp.getOutput();
    const bool isOrder1 = (mulMeanRsqrtOp.getInput1() == rsqrtResult && mulMeanRsqrtOp.getInput2() == meanResult);
    const bool isOrder2 = (mulMeanRsqrtOp.getInput1() == meanResult && mulMeanRsqrtOp.getInput2() == rsqrtResult);
    if (!isOrder1 && !isOrder2) {
        return matchFailed(rewriter, sqDiffOp, "Multiply inputs are not (mean, rsqrt)");
    }

    // --- 9. Subtract(0, Multiply(mean, rsqrt)) = negated-mean term ---
    auto subZeroOp = mlir::dyn_cast<IE::SubtractOp>(*mulMeanRsqrtOp->getUsers().begin());
    if (subZeroOp == nullptr || !subZeroOp->hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "Expected single-use SubtractOp(0, mul_mean_rsqrt)");
    }
    if (subZeroOp.getInput2() != mulMeanRsqrtOp.getOutput()) {
        return matchFailed(rewriter, sqDiffOp, "SubtractOp port1 is not Multiply(mean, rsqrt)");
    }
    if (!isZeroSplatConst(subZeroOp.getInput1())) {
        return matchFailed(rewriter, sqDiffOp, "SubtractOp minuend is not a zero constant");
    }

    // --- 10. Reshape/AffineReshape the negated-mean term for broadcasting ---
    auto* subZeroUser = *subZeroOp->getUsers().begin();
    if (!mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp>(subZeroUser) || !subZeroUser->hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "Expected single-use Reshape/AffineReshape after SubtractOp");
    }

    // --- 11. Walk forward through shape-only ops (Reshape/AffineReshape/Transpose) after the
    //         rsqrt-reshape head to find Multiply(x, rsqrt_r). An optional Transpose between
    //         the Reshape and the Multiply is skipped transparently. ---
    const mlir::Value mulXInput = skipShapeOpsForward(reshapeRsqrtHeadOp->getResult(0));
    if (!mulXInput.hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "Shape-chain after Reshape(rsqrt) has multiple uses");
    }
    auto mulXOp = mlir::dyn_cast<IE::MultiplyOp>(*mulXInput.getUsers().begin());
    if (mulXOp == nullptr || !mulXOp->hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "Expected single-use MultiplyOp(x, Reshape(rsqrt))");
    }
    const mlir::Value xBase = skipShapeOpsBackward(mvnInput);
    if (skipShapeOpsBackward(mulXOp.getInput1()) != xBase && skipShapeOpsBackward(mulXOp.getInput2()) != xBase) {
        return matchFailed(rewriter, sqDiffOp, "MultiplyOp(x, rsqrt_r): no input traces back to x");
    }

    // --- 12. Walk forward through shape-only ops after the neg-mean reshape to find the final Add.
    //         An optional Transpose between the Reshape and the Add is skipped transparently. ---
    const mlir::Value addFinalInput = skipShapeOpsForward(subZeroUser->getResult(0));
    if (!addFinalInput.hasOneUse()) {
        return matchFailed(rewriter, sqDiffOp, "Shape-chain after neg-mean Reshape has multiple uses");
    }
    auto addFinalOp = mlir::dyn_cast<IE::AddOp>(*addFinalInput.getUsers().begin());
    if (addFinalOp == nullptr) {
        return matchFailed(rewriter, sqDiffOp, "Expected AddOp as final MVN output op");
    }
    if (addFinalOp.getInput1() != mulXOp.getOutput() && addFinalOp.getInput2() != mulXOp.getOutput()) {
        return matchFailed(rewriter, sqDiffOp, "Final AddOp does not consume Multiply(x, Reshape(rsqrt))");
    }

    // --- 13. Resolve MVN1 shape mapping (with optional Transpose for non-reshape-friendly axes) ---
    normalizeAndSortAxes(inputMeanAxes, static_cast<int64_t>(inputRank));
    const auto mappingOpt = getMVN1Mapping(inputShape, inputMeanAxes);
    if (!mappingOpt.has_value()) {
        return matchFailed(rewriter, sqDiffOp, "Cannot reshape to MVN1-compatible shape");
    }
    const auto& mapping = mappingOpt.value();

    // --- 14. Emit [pre-transpose →] reshape → MVN → reshape [→ post-transpose], replace addFinalOp ---
    const auto ctx = sqDiffOp.getContext();
    const auto loc = sqDiffOp.getLoc();

    // Optional pre-transpose: required when a pure reshape cannot bring the
    // normalization axes into contiguous MVN spatial positions (e.g. CxHxW, axes [0,1]).
    mlir::Value mvnFeedVal = mvnInput;
    if (!mapping.preTransposePerm.empty()) {
        const auto preOrder =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(mapping.preTransposePerm, ctx));
        mvnFeedVal = rewriter.create<IE::TransposeOp>(appendLoc(loc, "pre_transpose"), mvnFeedVal, nullptr, preOrder)
                             .getOutput();
    }

    auto preReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "in_reshape"), mvnFeedVal,
                                                       getIntArrayAttr(ctx, mapping.mvnShape));

    const auto normVarianceAttr = mlir::BoolAttr::get(ctx, true);
    const auto acrossChannelsAttr = mlir::BoolAttr::get(ctx, mapping.isAcrossChannel);
    const auto epsAttr = getFPAttr(ctx, epsOpt.value());
    auto mvnOp = rewriter.create<IE::MVNOp>(appendLoc(loc, "mvn"), preReshapeOp.getOutput(), acrossChannelsAttr,
                                            normVarianceAttr, epsAttr);

    _log.trace("Replace '{0}' with MVN op '{1}'", loc, mvnOp);

    if (mapping.postTransposePerm.empty()) {
        // No transpose needed: reshape MVN output directly back to the original shape.
        auto outReshapeOp = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(
                addFinalOp, mvnOp.getOutput(), getIntArrayAttr(ctx, getShape(addFinalOp.getOutput())));
        extendOpLoc(outReshapeOp, "out_reshape");
    } else {
        // Transpose case: reshape to intermediate shape, then transpose back.
        auto outReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "out_reshape"), mvnOp.getOutput(),
                                                           getIntArrayAttr(ctx, mapping.postReshapeShape));

        const auto postOrder =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(mapping.postTransposePerm, ctx));
        auto postTransposeOp = rewriter.create<IE::TransposeOp>(appendLoc(loc, "post_transpose"),
                                                                outReshapeOp.getOutput(), nullptr, postOrder);
        rewriter.replaceOp(addFinalOp, postTransposeOp.getOutput());
    }
    return mlir::success();
}

//
// MVNFusionPass
//

class MVNFusionPass final : public IE::impl::MVNFusionBase<MVNFusionPass> {
public:
    explicit MVNFusionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MVNFusionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MVNFusion>(&ctx, _log);
    patterns.add<MVNFusionOvRef>(&ctx, _log);
    patterns.add<MVNFusionWithSquaredDiff>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createMVNFusionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMVNFusionPass(Logger log) {
    return std::make_unique<MVNFusionPass>(log);
}
