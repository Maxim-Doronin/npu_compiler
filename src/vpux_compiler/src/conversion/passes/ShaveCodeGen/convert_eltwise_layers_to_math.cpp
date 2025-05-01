//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Pass/Pass.h>

// Generated
namespace ConvertEltwiseLayersToMathPatterns {
#include <vpux/compiler/conversion/convert_eltwise_layers_to_math.hpp.inc>
}

namespace vpux {
#define GEN_PASS_DECL_CONVERTELTWISELAYERS2MATH
#define GEN_PASS_DEF_CONVERTELTWISELAYERS2MATH
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// ConvertSWLayers2LinalgPass
//

class ConvertEltwiseLayers2MathPass final : public impl::ConvertEltwiseLayers2MathBase<ConvertEltwiseLayers2MathPass> {
public:
    explicit ConvertEltwiseLayers2MathPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <typename SrcOp>
mlir::Value emitLinalgRegion(SrcOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(op);
    VPUX_UNUSED(args);
    VPUX_UNUSED(resultTypes);
    VPUX_UNUSED(rewriter);
    VPUX_THROW("emitLinalgRegion not specialized for operator");
}

// Bitwise layers

template <>
mlir::Value emitLinalgRegion<IE::BitwiseAndOp>(IE::BitwiseAndOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::arith::AndIOp>(op->getLoc(), resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::BitwiseOrOp>(IE::BitwiseOrOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::arith::OrIOp>(op->getLoc(), resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::BitwiseXorOp>(IE::BitwiseXorOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    return rewriter.create<mlir::arith::XOrIOp>(op->getLoc(), resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::BitwiseNotOp>(IE::BitwiseNotOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    // Do this as a xor with ~0 since there is no bitwise not operator in arith.
    auto allOnesAttr =
            rewriter.getIntegerAttr(resultTypes[0], llvm::APInt::getAllOnes(resultTypes[0].getIntOrFloatBitWidth()));
    auto allOnes = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), allOnesAttr);
    return rewriter.create<mlir::arith::XOrIOp>(op->getLoc(), resultTypes, args[0], allOnes);
}

// Logical and select layers

static mlir::Value emitNotEqualZero(mlir::Value val, mlir::PatternRewriter& rewriter) {
    // Emit a comparison of val to zero to determine its truthfulness (true if the
    // value is not equal to zero). This is used in the implementation of logical and
    // select layers for converting input values to bool (i1).
    if (mlir::isa<mlir::FloatType>(val.getType())) {
        auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.);
        mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);
        return rewriter.create<mlir::arith::CmpFOp>(val.getLoc(), mlir::arith::CmpFPredicate::ONE, val, zero,
                                                    mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
    }
    auto zeroAttr = rewriter.getIntegerAttr(val.getType(), 0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);

    return rewriter.create<mlir::arith::CmpIOp>(val.getLoc(), mlir::arith::CmpIPredicate::ne, val, zero);
}

static mlir::Value convertFromBool(mlir::Value val, mlir::Type ty, mlir::PatternRewriter& rewriter) {
    if (mlir::isa<mlir::FloatType>(ty)) {
        return rewriter.create<mlir::arith::UIToFPOp>(val.getLoc(), ty, val);
    }
    return rewriter.create<mlir::arith::ExtUIOp>(val.getLoc(), ty, val);
}

template <>
mlir::Value emitLinalgRegion<IE::SelectOp>(IE::SelectOp op, mlir::ValueRange args,
                                           llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto compare = emitNotEqualZero(args[0], rewriter);
    return rewriter.create<mlir::arith::SelectOp>(op->getLoc(), resultTypes[0], compare, args[1], args[2]);
}

template <>
mlir::Value emitLinalgRegion<IE::LogicalNotOp>(IE::LogicalNotOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    auto compare = emitNotEqualZero(args[0], rewriter);
    auto oneAttr = rewriter.getIntegerAttr(compare.getType(), 1);
    mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), oneAttr);  // i1 1
    auto logicalRes = rewriter.create<mlir::arith::XOrIOp>(op->getLoc(), compare, one);
    return convertFromBool(logicalRes, resultTypes[0], rewriter);
}

template <typename LogicalOp>
mlir::Value emitEltwiseLogical(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                               mlir::PatternRewriter& rewriter) {
    auto lhsCompare = emitNotEqualZero(args[0], rewriter);
    auto rhsCompare = emitNotEqualZero(args[1], rewriter);
    auto logicalRes = rewriter.create<LogicalOp>(op->getLoc(), lhsCompare, rhsCompare);
    return convertFromBool(logicalRes, resultTypes[0], rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::LogicalOrOp>(IE::LogicalOrOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    return emitEltwiseLogical<mlir::arith::OrIOp>(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::AndOp>(IE::AndOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return emitEltwiseLogical<mlir::arith::AndIOp>(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::LogicalXorOp>(IE::LogicalXorOp op, mlir::ValueRange args,
                                               llvm::ArrayRef<mlir::Type> resultTypes,
                                               mlir::PatternRewriter& rewriter) {
    return emitEltwiseLogical<mlir::arith::XOrIOp>(op, args, resultTypes, rewriter);
}

// Divide layers

template <>
mlir::Value emitLinalgRegion<IE::DivideOp>(IE::DivideOp op, mlir::ValueRange args,
                                           llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(elTy)) {
        // Add the arcp fastmath flag to allow converting to a multiplication
        // with the reciprocal. This matters for broadcasting (allows converting
        // the division to a multiplication when doing an inner broadcast).
        // SW layers also make (explicit) use of this.
        auto attr = mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), mlir::arith::FastMathFlags::arcp);
        return rewriter.create<mlir::arith::DivFOp>(loc, resultTypes, args[0], args[1], attr);
    }
    if (elTy.isSignedInteger()) {
        return rewriter.create<mlir::arith::DivSIOp>(loc, resultTypes, args);
    }
    return rewriter.create<mlir::arith::DivUIOp>(loc, resultTypes, args);
}

// Min/Max layers

template <>
mlir::Value emitLinalgRegion<IE::MaximumOp>(IE::MaximumOp op, mlir::ValueRange args,
                                            llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(elTy)) {
        // The sw layer implementation doesn't handle NaNs or signed zeroes,
        // so it should be okay to have nnan/nsz.
        auto attr = mlir::arith::FastMathFlagsAttr::get(
                rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
        return rewriter.create<mlir::arith::MaximumFOp>(loc, resultTypes, args[0], args[1], attr);
    }
    if (elTy.isSignedInteger()) {
        return rewriter.create<mlir::arith::MaxSIOp>(loc, resultTypes, args);
    }
    return rewriter.create<mlir::arith::MaxUIOp>(loc, resultTypes, args);
}

template <>
mlir::Value emitLinalgRegion<IE::MinimumOp>(IE::MinimumOp op, mlir::ValueRange args,
                                            llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(elTy)) {
        // Same as for maximum, we can add nnan/nsz.
        auto attr = mlir::arith::FastMathFlagsAttr::get(
                rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
        return rewriter.create<mlir::arith::MinimumFOp>(loc, resultTypes, args[0], args[1], attr);
    }
    if (elTy.isSignedInteger()) {
        return rewriter.create<mlir::arith::MinSIOp>(loc, resultTypes, args);
    }
    return rewriter.create<mlir::arith::MinUIOp>(loc, resultTypes, args);
}

template <typename SrcOp>
class IEEltwiseToLinalg : public mlir::OpRewritePattern<SrcOp> {
public:
    using mlir::OpRewritePattern<SrcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SrcOp op, mlir::PatternRewriter& rewriter) const final {
        auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResultTypes().front());
        auto resultElTy = mlir::cast<vpux::NDTypeInterface>(resultType).getElementType();
        auto outputShape = mlir::cast<vpux::NDTypeInterface>(op->getResultTypes().front()).getShape();

        auto getElTyAsSignlessIfInt = [&](mlir::Type ty) {
            assert(mlir::isa<mlir::RankedTensorType>(ty) && "Ranked tensor type required for getElTyAsSignlessIfInt");
            // Get the math/arith compatible element type for the ty tensor type.
            // math/arith dialects don't accept non-signless integers.
            auto elTy = mlir::cast<vpux::NDTypeInterface>(ty).getElementType();
            auto signlessElTy = mlir::isa<mlir::IntegerType>(elTy) && !elTy.isSignlessInteger()
                                        ? mlir::IntegerType::get(rewriter.getContext(), getElemTypeSize(elTy).count())
                                        : elTy;
            return signlessElTy;
        };

        auto linalgResultElTy = getElTyAsSignlessIfInt(resultType);
        bool allowBroadcast = false;
        if (op->getOperands().size() > 1) {
            // TODO: E#159770 - there should be a broadcastable op interface.
            if (auto broadcastAddr = op->template getAttrOfType<IE::AutoBroadcastTypeAttr>("auto_broadcast")) {
                switch (broadcastAddr.getValue()) {
                case IE::AutoBroadcastType::NONE_OR_EXPLICIT:
                    break;
                case IE::AutoBroadcastType::NUMPY:
                    allowBroadcast = true;
                    break;
                default:
                    // We don't support paddle paddle broadcasting.
                    return mlir::failure();
                }
            }
        }
        if (allowBroadcast) {
            // Reject the dynamic output type, at least for now.
            // We could support more cases though, at least where there's no ambiguity as to where the
            // broadcast is coming from (e.g <1 x ?>, <4 x 1> -> <4, ?>).
            if (outputShape.isDynamic()) {
                return mlir::failure();
            }
        }

        // Create the linalg affine maps. This will handle broadcasting (operand dimension size
        // is equal to 1 and is different than the result dimension) by using a 0 affine constant
        // in the affine map.
        auto rank = resultType.getRank();
        bool hasBroadcast = false;
        auto affineMaps = llvm::map_to_vector(op->getOperands(), [&](mlir::Value operand) {
            auto shape = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getShape();
            SmallVector<mlir::AffineExpr> affineExprs;
            auto operandRank = mlir::cast<mlir::RankedTensorType>(operand.getType()).getRank();
            // The output rank should always be larger or equal than the operand rank
            // due to shape inference rules.
            assert(rank >= operandRank && "Unexpected input rank");
            for (auto it : llvm::enumerate(shape)) {
                // Match the input dimension to the output dimension index according
                // to numpy broadcasting rules. Note if the input and output shapes
                // have different ranks then the lower ranked shape (the input one)
                // is right aligned and filled with ones to the left to equalize
                // the ranks.
                auto outIdx = rank - operandRank + it.index();
                auto outDim = outputShape.raw()[outIdx];
                // If the input dimension is equal to one and the output dimension is
                // not one then we are broadcasting.
                auto broadcastDim = allowBroadcast && it.value() == 1 && outDim != it.value();
                // Now that we've figured out if we are broadcasting or not we can
                // update the overall broadcast flag.
                hasBroadcast = hasBroadcast || broadcastDim;
                // Broadcasting across this dimension is equivalent to having a constant
                // zero expression in the affine map.
                auto affineExpr = broadcastDim ? rewriter.getAffineConstantExpr(0) : rewriter.getAffineDimExpr(outIdx);
                affineExprs.push_back(affineExpr);
            }
            return mlir::AffineMap::get(rank, 0, affineExprs, rewriter.getContext());
        });
        // Add the affine map for the output tensor as well.
        affineMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

        auto linalgOperands = llvm::map_to_vector(op->getOperands(), [&](mlir::Value operand) {
            auto elTy = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getElementType();
            auto signlessElTy = getElTyAsSignlessIfInt(operand.getType());
            if (elTy == signlessElTy) {
                // No cast required.
                return operand;
            }
            // The input type is a signed/unsigned integer so not compatible
            // we the dialects we need for lowering. We need to bitcast this
            // to a signless integer type.
            auto outputTy = mlir::cast<vpux::NDTypeInterface>(operand.getType()).changeElemType(signlessElTy);
            mlir::Value castResult =
                    rewriter.create<mlir::tensor::BitcastOp>(op.getLoc(), outputTy, operand).getResult();
            return castResult;
        });

        // We need a tensor::EmptyOp to cover the case where the output tensor type is different than the
        // input ones. This can be caused either by broadcasting (the shape changes) or if we get
        // a different element type for the output (bitcasting is possibly illegal).
        // This should be removed after outlining.
        auto inputElTy = mlir::cast<vpux::NDTypeInterface>(linalgOperands.front().getType()).getElementType();
        bool changesType = (inputElTy != linalgResultElTy);
        mlir::Value outputTensor = hasBroadcast || changesType ? rewriter.create<mlir::tensor::EmptyOp>(
                                                                         op.getLoc(), outputShape, linalgResultElTy)
                                                               : linalgOperands.front();

        llvm::SmallVector<mlir::utils::IteratorType> loopAttrs(rank, mlir::utils::IteratorType::parallel);
        auto linalgOp = rewriter.create<mlir::linalg::GenericOp>(
                op.getLoc(), outputTensor.getType(), linalgOperands, outputTensor, affineMaps, loopAttrs,
                [&](mlir::OpBuilder& opBuilder, mlir::Location loc, mlir::ValueRange blockArgs) {
                    mlir::Value opResult = emitLinalgRegion<SrcOp>(op, blockArgs.take_front(op->getNumOperands()),
                                                                   {linalgResultElTy}, rewriter);
                    opBuilder.create<mlir::linalg::YieldOp>(loc, opResult);
                });

        if (linalgResultElTy != resultElTy) {
            // Bitcast the result back to the original type.
            auto castOp = rewriter.create<mlir::tensor::BitcastOp>(op.getLoc(), resultType, linalgOp.getResult(0))
                                  .getResult();
            rewriter.replaceOp(op, castOp);
            return mlir::success();
        }

        rewriter.replaceOp(op, linalgOp);
        return mlir::success();
    }
};

void ConvertEltwiseLayers2MathPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                           mlir::math::MathDialect, mlir::func::FuncDialect>();

    target.addLegalDialect<IE::IEDialect>();
    target.addIllegalOp<IE::CosOp>();
    target.addIllegalOp<IE::MaximumOp>();
    target.addIllegalOp<IE::MinimumOp>();
    target.addIllegalOp<IE::DivideOp>();
    target.addIllegalOp<IE::LogOp>();
    target.addIllegalOp<IE::ExpOp>();
    target.addIllegalOp<IE::BitwiseAndOp, IE::BitwiseOrOp, IE::BitwiseXorOp, IE::BitwiseNotOp>();
    target.addIllegalOp<IE::LogicalOrOp, IE::LogicalXorOp, IE::AndOp, IE::LogicalNotOp, IE::SelectOp>();

    mlir::RewritePatternSet patterns(&ctx);
    ConvertEltwiseLayersToMathPatterns::populateWithGenerated(patterns);
    patterns.add<IEEltwiseToLinalg<IE::MaximumOp>, IEEltwiseToLinalg<IE::MinimumOp>, IEEltwiseToLinalg<IE::DivideOp>>(
            &ctx);
    patterns.add<IEEltwiseToLinalg<IE::BitwiseAndOp>, IEEltwiseToLinalg<IE::BitwiseOrOp>,
                 IEEltwiseToLinalg<IE::BitwiseXorOp>, IEEltwiseToLinalg<IE::BitwiseNotOp>>(&ctx);
    patterns.add<IEEltwiseToLinalg<IE::LogicalOrOp>, IEEltwiseToLinalg<IE::LogicalXorOp>, IEEltwiseToLinalg<IE::AndOp>,
                 IEEltwiseToLinalg<IE::LogicalNotOp>, IEEltwiseToLinalg<IE::SelectOp>>(&ctx);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertEltwiseLayers2MathPass
//

std::unique_ptr<mlir::Pass> ShaveCodeGen::createConvertEltwiseLayers2MathPass(Logger log) {
    return std::make_unique<ConvertEltwiseLayers2MathPass>(log);
}
