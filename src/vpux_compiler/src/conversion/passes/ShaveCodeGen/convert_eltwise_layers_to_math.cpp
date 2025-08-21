//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

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

// Comparison layers

template <mlir::arith::CmpIPredicate signedP, mlir::arith::CmpIPredicate unsignedP, mlir::arith::CmpFPredicate fpP>
static mlir::Value emitCmp(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                           mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    mlir::Value cmpVal = nullptr;
    if (mlir::isa<mlir::FloatType>(elTy)) {
        cmpVal = rewriter.create<mlir::arith::CmpFOp>(
                op->getLoc(), fpP, args[0], args[1],
                mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);
    } else if (elTy.isSignedInteger()) {
        cmpVal = rewriter.create<mlir::arith::CmpIOp>(op->getLoc(), signedP, args[0], args[1]);
    } else {
        cmpVal = rewriter.create<mlir::arith::CmpIOp>(op->getLoc(), unsignedP, args[0], args[1]);
    }

    if (mlir::isa<mlir::FloatType>(resultTypes[0])) {
        return rewriter.create<mlir::arith::UIToFPOp>(op->getLoc(), resultTypes[0], cmpVal);
    }

    return rewriter.create<mlir::arith::ExtUIOp>(op->getLoc(), resultTypes[0], cmpVal);
}

template <>
mlir::Value emitLinalgRegion(IE::EqualOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::eq, mlir::arith::CmpIPredicate::eq, mlir::arith::CmpFPredicate::OEQ>(
            op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion(IE::NotEqualOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::ne, mlir::arith::CmpIPredicate::ne, mlir::arith::CmpFPredicate::ONE>(
            op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion(IE::LessOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::slt, mlir::arith::CmpIPredicate::ult, mlir::arith::CmpFPredicate::OLT>(
            op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion(IE::LessEqualOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::sle, mlir::arith::CmpIPredicate::ule, mlir::arith::CmpFPredicate::OLE>(
            op, args, resultTypes, rewriter);
}
template <>
mlir::Value emitLinalgRegion(IE::GreaterOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::sgt, mlir::arith::CmpIPredicate::ugt, mlir::arith::CmpFPredicate::OGT>(
            op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion(IE::GreaterEqualOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                             mlir::PatternRewriter& rewriter) {
    return emitCmp<mlir::arith::CmpIPredicate::sge, mlir::arith::CmpIPredicate::uge, mlir::arith::CmpFPredicate::OGE>(
            op, args, resultTypes, rewriter);
}

// Squared diff op

template <>
mlir::Value emitLinalgRegion<IE::SquaredDifferenceOp>(IE::SquaredDifferenceOp op, mlir::ValueRange args,
                                                      llvm::ArrayRef<mlir::Type> resultTypes,
                                                      mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    if (mlir::isa<mlir::FloatType>(args[0].getType())) {
        auto sub = rewriter.create<mlir::arith::SubFOp>(loc, resultTypes, args);
        return rewriter.create<mlir::arith::MulFOp>(loc, sub.getType(), sub, sub);
    }
    auto sub = rewriter.create<mlir::arith::SubIOp>(loc, resultTypes, args);
    return rewriter.create<mlir::arith::MulIOp>(loc, sub.getType(), sub, sub);
}

// Clamp/ReLU/LeakyRelu

static mlir::Value emitClamp(mlir::Value val, mlir::Type elTy, double min, double max,
                             mlir::PatternRewriter& rewriter) {
    auto loc = val.getLoc();
    if (mlir::isa<mlir::IntegerType>(val.getType())) {
        auto bw = val.getType().getIntOrFloatBitWidth();
        bool isExact;
        bool isSigned = elTy.isSignedInteger();
        llvm::APFloat minF(min), maxF(max);
        llvm::APSInt minI(bw, /*isUnsigned=*/!isSigned);
        llvm::APSInt maxI(bw, /*isUnsigned=*/!isSigned);

        // From the openvino documentation:
        // Note: In case of integral numeric type, ceil is used to convert min from float to T and floor is used to
        // convert max from float to T.
        //
        // Convert the min/max values to the specified integer type. convertToInt
        // will give us MIN_INT or MAX_INT if the value in case of underflow or overflow.
        minF.convertToInteger(minI, llvm::APFloat::rmTowardPositive, &isExact);
        maxF.convertToInteger(maxI, llvm::APFloat::rmTowardNegative, &isExact);

        // Construct the min/max constants.
        auto minCt = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(val.getType(), minI));
        auto maxCt = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(val.getType(), maxI));

        auto cmpPred = isSigned ? mlir::arith::CmpIPredicate::sle : mlir::arith::CmpIPredicate::ule;

        // Compute max(input, min_limit).
        auto cmpMin = rewriter.create<mlir::arith::CmpIOp>(loc, cmpPred, val, minCt);
        auto lowClamped = rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmpMin, minCt, val);

        // Compute min(max(input, min_limit), max_limit).
        auto cmpMax = rewriter.create<mlir::arith::CmpIOp>(loc, cmpPred, maxCt, lowClamped);
        return rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmpMax, maxCt, lowClamped);
    }

    // Construct the min/max constants.
    auto minCt = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(val.getType(), min));
    auto maxCt = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(val.getType(), max));
    auto fmFlags = mlir::arith::FastMathFlags::nsz | mlir::arith::FastMathFlags::nnan;

    // Compute max(input, min_limit).
    auto cmpMin = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, val, minCt, fmFlags);
    auto lowClamped = rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmpMin, minCt, val);

    // Compute min(max(input, min_limit), max_limit).
    auto cmpMax =
            rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lowClamped, maxCt, fmFlags);
    return rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmpMax, lowClamped, maxCt);
}

template <>
mlir::Value emitLinalgRegion<IE::ClampOp>(IE::ClampOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    return emitClamp(args[0], elTy, op.getMinAttr().getValueAsDouble(), op.getMaxAttr().getValueAsDouble(), rewriter);
}

static mlir::Value emitLeakyReLU(mlir::Value val, double slope, mlir::PatternRewriter& rewriter) {
    // Compute this as max(in, 0) + slope * min(in, 0)
    auto loc = val.getLoc();
    auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);

    auto cmp = rewriter.create<mlir::arith::CmpFOp>(val.getLoc(), mlir::arith::CmpFPredicate::OLE, val, zero,
                                                    mlir::arith::FastMathFlags::nsz | mlir::arith::FastMathFlags::nnan);
    auto max = rewriter.create<mlir::arith::SelectOp>(loc, val.getType(), cmp, zero, val);

    if (slope == 0.) {
        return max;
    }

    auto min = rewriter.create<mlir::arith::SelectOp>(loc, cmp, val, zero);
    auto slopeAttr = rewriter.getFloatAttr(val.getType(), slope);
    auto slopeCt = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), slopeAttr);
    auto mul = rewriter.create<mlir::arith::MulFOp>(loc, min, slopeCt);
    return rewriter.create<mlir::arith::AddFOp>(loc, max, mul);
}

template <>
mlir::Value emitLinalgRegion<IE::LeakyReluOp>(IE::LeakyReluOp op, mlir::ValueRange args,
                                              llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    return emitLeakyReLU(args[0], op.getNegativeSlopeAttr().getValueAsDouble(), rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::ReLUOp>(IE::ReLUOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    VPUX_UNUSED(op);
    return emitLeakyReLU(args[0], 0., rewriter);
}

// Add/Sub/Mul

template <typename SrcIOp, typename SrcFOp>
mlir::Value emitEltwiseWithPostOp(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                  mlir::PatternRewriter& rewriter) {
    mlir::Value val = nullptr;
    if (mlir::isa<mlir::FloatType>(args[0].getType())) {
        val = rewriter.create<SrcFOp>(op->getLoc(), resultTypes, args);
    } else {
        val = rewriter.create<SrcIOp>(op->getLoc(), resultTypes, args);
    }
    if (auto postOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op)) {
        if (auto postOpAttr = postOp.getPostOp()) {
            val = llvm::TypeSwitch<mlir::Attribute, mlir::Value>(postOpAttr)
                          .Case<IE::ReluAttr>([&](auto attr) {
                              VPUX_UNUSED(attr);
                              return emitLeakyReLU(val, 0., rewriter);
                          })
                          .template Case<IE::LeakyReluAttr>([&](auto attr) {
                              VPUX_UNUSED(attr);
                              return emitLeakyReLU(val, attr.getNegativeSlope().getValueAsDouble(), rewriter);
                          })
                          .Default([&](auto attr) -> mlir::Value {
                              VPUX_UNUSED(attr);
                              VPUX_THROW("Unsupported postop for operation '{0}' at '{1}'", op->getName(),
                                         op->getLoc());
                          });
        }
    }
    return val;
}

template <>
mlir::Value emitLinalgRegion<IE::MultiplyOp>(IE::MultiplyOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    assert(op.getClampAttr() == nullptr && "Unexpected clamp attribute");
    return emitEltwiseWithPostOp<mlir::arith::MulIOp, mlir::arith::MulFOp>(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::SubtractOp>(IE::SubtractOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    assert(op.getClampAttr() == nullptr && "Unexpected clamp attribute");
    return emitEltwiseWithPostOp<mlir::arith::SubIOp, mlir::arith::SubFOp>(op, args, resultTypes, rewriter);
}

template <>
mlir::Value emitLinalgRegion<IE::AddOp>(IE::AddOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    assert(op.getClampAttr() == nullptr && "Unexpected clamp attribute");
    return emitEltwiseWithPostOp<mlir::arith::AddIOp, mlir::arith::AddFOp>(op, args, resultTypes, rewriter);
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

template <typename OpT>
mlir::Value emitF16ViaF32ThenTrunc(mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                   mlir::PatternRewriter& rewriter) {
    auto elTy = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getElementType();

    // In case the type is f16 then we need to do a conversion to f32 and convert the result back to f16
    if (mlir::isa<mlir::Float16Type>(elTy)) {
        // f16 -> fp32
        auto f32Type = mlir::FloatType::getF32(rewriter.getContext());
        auto extArg = rewriter.create<mlir::arith::ExtFOp>(args[0].getLoc(), f32Type, args[0]);

        // Creating the operation in fp32
        auto genOp = rewriter.create<OpT>(op->getLoc(), f32Type, extArg);
        auto f16Type = mlir::FloatType::getF16(rewriter.getContext());
        return rewriter.create<mlir::arith::TruncFOp>(op->getLoc(), f16Type, genOp);
    }

    // Non-F16: direct call operation
    return rewriter.create<OpT>(op->getLoc(), resultTypes, args[0]);
}

// Erf layer
template <>
mlir::Value emitLinalgRegion<IE::ErfOp>(IE::ErfOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return emitF16ViaF32ThenTrunc<mlir::math::ErfOp>(op, args, resultTypes, rewriter);
}

// Round layer
template <>
mlir::Value emitLinalgRegion<IE::RoundOp>(IE::RoundOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    auto modeAttr = op.getMode();

    if (modeAttr == IE::RoundMode::HALF_TO_EVEN) {
        return rewriter.create<mlir::math::RoundEvenOp>(loc, resultTypes, args[0]);
    }
    return rewriter.create<mlir::math::RoundOp>(loc, resultTypes, args[0]);
}

// Sin layer
template <>
mlir::Value emitLinalgRegion<IE::SinOp>(IE::SinOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return emitF16ViaF32ThenTrunc<mlir::math::SinOp>(op, args, resultTypes, rewriter);
}

// Cos layer
template <>
mlir::Value emitLinalgRegion<IE::CosOp>(IE::CosOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    return emitF16ViaF32ThenTrunc<mlir::math::CosOp>(op, args, resultTypes, rewriter);
}

// Cosh layer   cosh(x)=0.5×(exp(X)+exp(-X))

template <>
mlir::Value emitLinalgRegion<IE::CoshOp>(IE::CoshOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto input = args.front();
    auto zeroConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.0));  // Const 0
    auto halfConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.5));  // Const 0.5
    auto xExp = rewriter.create<mlir::math::ExpOp>(loc, input, mlir::arith::FastMathFlags::afn);         // exp(X)
    auto negX = rewriter.create<mlir::arith::SubFOp>(loc, zeroConst, input);                             // -x = 0-x
    auto expNegX = rewriter.create<mlir::math::ExpOp>(loc, negX, mlir::arith::FastMathFlags::afn);       // exp(-x)
    auto expSum = rewriter.create<mlir::arith::AddFOp>(loc, xExp, expNegX);  // exp(x) + exp(-x)

    return rewriter.create<mlir::arith::MulFOp>(loc, expSum, halfConst);
}

// tan(x) = sin(x) / cos(x)

template <>
mlir::Value emitLinalgRegion<IE::TanOp>(IE::TanOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto sinX = emitF16ViaF32ThenTrunc<mlir::math::SinOp>(op, args, resultTypes, rewriter);
    auto cosX = emitF16ViaF32ThenTrunc<mlir::math::CosOp>(op, args, resultTypes, rewriter);

    return rewriter.create<mlir::arith::DivFOp>(loc, sinX, cosX);
}

// atanh(x) = 0.5 * log((1 + x) / (1 - x))

template <>
mlir::Value emitLinalgRegion<IE::AtanhOp>(IE::AtanhOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                          mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto input = args.front();

    auto oneConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 1.0));  // Const 1
    auto halfConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.5));  // Const 0.5

    auto onePlusX = rewriter.create<mlir::arith::AddFOp>(loc, oneConst, input);   // 1 + x
    auto oneMinusX = rewriter.create<mlir::arith::SubFOp>(loc, oneConst, input);  // 1 - x

    auto divResult = rewriter.create<mlir::arith::DivFOp>(loc, onePlusX, oneMinusX);  // (1 + x) / (1 - x)
    auto logResult = rewriter.create<mlir::math::LogOp>(loc, divResult,
                                                        mlir::arith::FastMathFlags::afn);  // log((1 + x) / (1 - x))

    return rewriter.create<mlir::arith::MulFOp>(loc, logResult, halfConst);
}

// Sinh layer   sinh(x) = (exp(x) - exp(-x)) / 2

template <>
mlir::Value emitLinalgRegion<IE::SinhOp>(IE::SinhOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    VPUX_UNUSED(resultTypes);
    auto loc = op->getLoc();
    auto input = args.front();
    auto zeroConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.0));  // Const 0
    auto halfConst =
            rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(input.getType(), 0.5));  // Const 0.5
    auto expX = rewriter.create<mlir::math::ExpOp>(loc, input, mlir::arith::FastMathFlags::afn);         // exp(x)
    auto negX = rewriter.create<mlir::arith::SubFOp>(loc, zeroConst, input);                             // -x = 0-x
    auto expNegX = rewriter.create<mlir::math::ExpOp>(loc, negX, mlir::arith::FastMathFlags::afn);       // exp (-x)
    auto diffExp = rewriter.create<mlir::arith::SubFOp>(loc, expX, expNegX);  // exp(x) - exp(-x)

    return rewriter.create<mlir::arith::MulFOp>(loc, diffExp, halfConst);
}

// Negative layer
template <>
mlir::Value emitLinalgRegion<IE::NegativeOp>(IE::NegativeOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto loc = op.getLoc();

    if (mlir::isa<mlir::FloatType>(args[0].getType())) {
        auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(args[0].getType(), 0.));
        return rewriter.create<mlir::arith::SubFOp>(loc, resultTypes, mlir::ValueRange{zero, args[0]});
    }

    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(args[0].getType(), 0));
    return rewriter.create<mlir::arith::SubIOp>(loc, resultTypes, mlir::ValueRange{zero, args[0]});
}

// Callback type for emitting the linalg body for an operation.
using EmitBodyCallback = std::function<mlir::Value(mlir::Operation*, mlir::ValueRange, llvm::ArrayRef<mlir::Type>,
                                                   mlir::PatternRewriter&)>;

static mlir::LogicalResult emitLinalgHelper(mlir::Operation* op, EmitBodyCallback callback,
                                            mlir::PatternRewriter& rewriter) {
    auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResultTypes().front());
    auto outputShape = mlir::cast<vpux::NDTypeInterface>(op->getResultTypes().front()).getShape();
    auto linalgResultElTy = ShaveCodeGen::getLinalgElementType(resultType, rewriter.getContext());

    bool allowBroadcast = false;
    if (op->getOperands().size() > 1) {
        // TODO: E#159770 - there should be a broadcastable op interface.
        if (auto broadcastAddr = op->getAttrOfType<IE::AutoBroadcastTypeAttr>("auto_broadcast")) {
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
    auto inverseOutputMap = mlir::inversePermutation(mlir::cast<vpux::NDTypeInterface>(op->getResultTypes().front())
                                                             .getDimsOrder()
                                                             .toAffineMap(rewriter.getContext()));

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

        // Compose affine maps to get the correct indexing for this operand
        // considering that the output tensor will have identity indexing.
        auto opMap =
                mlir::cast<vpux::NDTypeInterface>(operand.getType()).getDimsOrder().toAffineMap(rewriter.getContext());
        auto logicalMap = mlir::AffineMap::get(rank, 0, affineExprs, rewriter.getContext());
        return opMap.compose(logicalMap).compose(inverseOutputMap);
    });

    // Add the affine map for the output tensor as well.
    affineMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    auto linalgOperands = llvm::map_to_vector(op->getOperands(), [&](mlir::Value operand) {
        return ShaveCodeGen::convertToLinalgValue(operand, rewriter);
    });

    // We need a tensor::EmptyOp to cover the case where the output tensor type is different than the
    // input ones. This can be caused either by broadcasting (the shape changes) or if we get
    // a different element type for the output (bitcasting is possibly illegal).
    // This should be removed after outlining.
    auto inputElTy = mlir::cast<vpux::NDTypeInterface>(linalgOperands.front().getType()).getElementType();
    bool changesType = (inputElTy != linalgResultElTy);
    mlir::Value outputTensor = nullptr;

    if (hasBroadcast || changesType || op->getOperand(0).getType() != op->getResult(0).getType()) {
        // Compute the tensor shape with an identity layout which has a memory layout that matches our
        // original output tensor.
        auto ndResultTy = mlir::cast<NDTypeInterface>(resultType);
        auto dOrder = DimsOrder::fromPermutation(ndResultTy.getDimsOrder().toPermutation());
        auto dstShape = dOrder.toMemoryOrder(ndResultTy.getShape()).raw();
        outputTensor = rewriter.create<mlir::tensor::EmptyOp>(op->getLoc(), dstShape, linalgResultElTy);
    } else {
        outputTensor = linalgOperands.front();
    }

    llvm::SmallVector<mlir::utils::IteratorType> loopAttrs(rank, mlir::utils::IteratorType::parallel);
    auto linalgOp = rewriter.create<mlir::linalg::GenericOp>(
            op->getLoc(), outputTensor.getType(), linalgOperands, outputTensor, affineMaps, loopAttrs,
            [&](mlir::OpBuilder& opBuilder, mlir::Location loc, mlir::ValueRange blockArgs) {
                mlir::Value opResult =
                        callback(op, blockArgs.take_front(op->getNumOperands()), {linalgResultElTy}, rewriter);
                opBuilder.create<mlir::linalg::YieldOp>(loc, opResult);
            });

    rewriter.replaceOp(
            op, ShaveCodeGen::convertFromLinalgValue(linalgOp->getResult(0), op->getResult(0).getType(), rewriter)
                        .getDefiningOp());
    return mlir::success();
}

template <typename SrcOp>
class IEEltwiseToLinalg : public mlir::OpRewritePattern<SrcOp> {
public:
    using mlir::OpRewritePattern<SrcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SrcOp op, mlir::PatternRewriter& rewriter) const final {
        auto emitBody = [](mlir::Operation* op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> types,
                           mlir::PatternRewriter& rewriter) {
            return emitLinalgRegion<SrcOp>(mlir::cast<SrcOp>(op), args, types, rewriter);
        };
        return emitLinalgHelper(op, emitBody, rewriter);
    }
};

// Convert layers

template <>
mlir::Value emitLinalgRegion<IE::ConvertOp>(IE::ConvertOp op, mlir::ValueRange args,
                                            llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    auto inputTy = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getElementType();
    auto resultType = mlir::cast<mlir::RankedTensorType>(op.getOutput().getType());
    auto outputTy = mlir::cast<vpux::NDTypeInterface>(resultType).getElementType();
    auto loc = op.getLoc();

    // Float to Int
    if (mlir::isa<mlir::FloatType>(inputTy) && mlir::isa<mlir::IntegerType>(outputTy)) {
        if (outputTy.isSignedInteger()) {
            return rewriter.create<mlir::arith::FPToSIOp>(loc, resultTypes, args);
        }

        return rewriter.create<mlir::arith::FPToUIOp>(loc, resultTypes, args);
    }

    // Int to Float
    if (mlir::isa<mlir::IntegerType>(inputTy) && mlir::isa<mlir::FloatType>(outputTy)) {
        if (inputTy.isSignedInteger()) {
            return rewriter.create<mlir::arith::SIToFPOp>(loc, resultTypes, args);
        }

        return rewriter.create<mlir::arith::UIToFPOp>(loc, resultTypes, args);
    }

    // Float to Float
    if (mlir::isa<mlir::FloatType>(inputTy) && mlir::isa<mlir::FloatType>(outputTy)) {
        if (inputTy == outputTy) {
            return args[0];
        }

        if (outputTy.getIntOrFloatBitWidth() > inputTy.getIntOrFloatBitWidth()) {
            return rewriter.create<mlir::arith::ExtFOp>(loc, resultTypes, args);
        }

        return rewriter.create<mlir::arith::TruncFOp>(loc, resultTypes, args);
    }

    // Int to Int
    if (mlir::isa<mlir::IntegerType>(inputTy) && mlir::isa<mlir::IntegerType>(outputTy)) {
        if (outputTy.getIntOrFloatBitWidth() == inputTy.getIntOrFloatBitWidth()) {
            return args[0];
        }

        if (outputTy.getIntOrFloatBitWidth() > inputTy.getIntOrFloatBitWidth()) {
            if (inputTy.isSignedInteger()) {
                return rewriter.create<mlir::arith::ExtSIOp>(loc, resultTypes, args);
            }

            return rewriter.create<mlir::arith::ExtUIOp>(loc, resultTypes, args);
        }

        if (outputTy.getIntOrFloatBitWidth() < inputTy.getIntOrFloatBitWidth()) {
            return rewriter.create<mlir::arith::TruncIOp>(loc, resultTypes, args);
        }
    }

    VPUX_THROW("Unsupported convert type combination");
}

// Abs layer
template <>
mlir::Value emitLinalgRegion<IE::AbsOp>(IE::AbsOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                        mlir::PatternRewriter& rewriter) {
    auto loc = op.getLoc();

    return rewriter.create<mlir::math::AbsFOp>(loc, resultTypes, args);
}

// Sign layer
template <>
mlir::Value emitLinalgRegion<IE::SignOp>(IE::SignOp op, mlir::ValueRange args, llvm::ArrayRef<mlir::Type> resultTypes,
                                         mlir::PatternRewriter& rewriter) {
    // This algorithm computes the sign of a floating-point value by examining its sign bit.
    // To handle the case where there is -0, we check if the value is exactly zero
    // (including both +0 and -0) and return 0 in that case.
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto val = args[0];

    auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);
    auto negAttr = rewriter.getFloatAttr(val.getType(), -1.0);
    mlir::Value negOne = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), negAttr);
    auto posAttr = rewriter.getFloatAttr(val.getType(), 1.0);
    mlir::Value posOne = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), posAttr);

    auto bitWidth = val.getType().getIntOrFloatBitWidth();
    auto intTy = mlir::IntegerType::get(val.getContext(), bitWidth);
    auto casted = rewriter.create<mlir::arith::BitcastOp>(loc, intTy, val);
    llvm::APInt msbMaskVal = llvm::APInt::getSignMask(bitWidth);
    auto msbMask = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(intTy, msbMaskVal));
    auto signBit = rewriter.create<mlir::arith::AndIOp>(loc, casted, msbMask);

    auto shift = rewriter.create<mlir::arith::ShLIOp>(loc, casted,
                                                      rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, intTy));
    auto isZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, shift,
                                                       rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intTy));

    auto signHandled = rewriter.create<mlir::arith::SelectOp>(
            loc,
            rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, signBit,
                                                 rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intTy)),
            negOne, posOne);

    return rewriter.create<mlir::arith::SelectOp>(loc, isZero, zero, signHandled);
}

// HSwish layer
template <>
mlir::Value emitLinalgRegion<IE::HSwishOp>(IE::HSwishOp op, mlir::ValueRange args,
                                           llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    // Compute this as x*((min(max(x+3,0),6))/6)
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto val = args[0];
    auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);
    auto threeAttr = rewriter.getFloatAttr(val.getType(), 3.0);
    mlir::Value three = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), threeAttr);
    auto sixAttr = rewriter.getFloatAttr(val.getType(), 6.0);
    mlir::Value six = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), sixAttr);
    auto divSixAttr = rewriter.getFloatAttr(val.getType(), 1.0 / 6.0);
    mlir::Value divSix = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), divSixAttr);
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(
            rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);

    auto add = rewriter.create<mlir::arith::AddFOp>(loc, val, three);
    auto max = rewriter.create<mlir::arith::MaximumFOp>(loc, add, zero, fmFlags);
    auto min = rewriter.create<mlir::arith::MinimumFOp>(loc, max, six, fmFlags);
    auto mul = rewriter.create<mlir::arith::MulFOp>(loc, min, divSix);

    return rewriter.create<mlir::arith::MulFOp>(loc, val, mul);
}

// HSigmoid Layer
template <>
mlir::Value emitLinalgRegion<IE::HSigmoidOp>(IE::HSigmoidOp op, mlir::ValueRange args,
                                             llvm::ArrayRef<mlir::Type> resultTypes, mlir::PatternRewriter& rewriter) {
    // Compute this as (min(max(x+3,0),6))/6
    VPUX_UNUSED(resultTypes);
    auto loc = op.getLoc();
    auto val = args[0];
    auto zeroAttr = rewriter.getFloatAttr(val.getType(), 0.0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), zeroAttr);
    auto threeAttr = rewriter.getFloatAttr(val.getType(), 3.0);
    mlir::Value three = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), threeAttr);
    auto sixAttr = rewriter.getFloatAttr(val.getType(), 6.0);
    mlir::Value six = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), sixAttr);
    auto divSixAttr = rewriter.getFloatAttr(val.getType(), 1.0 / 6.0);
    mlir::Value divSix = rewriter.create<mlir::arith::ConstantOp>(val.getLoc(), divSixAttr);
    auto fmFlags = mlir::arith::FastMathFlagsAttr::get(
            rewriter.getContext(), mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::nsz);

    auto add = rewriter.create<mlir::arith::AddFOp>(loc, val, three);
    auto max = rewriter.create<mlir::arith::MaximumFOp>(loc, add, zero, fmFlags);
    auto min = rewriter.create<mlir::arith::MinimumFOp>(loc, max, six, fmFlags);

    return rewriter.create<mlir::arith::MulFOp>(loc, min, divSix);
}

void ConvertEltwiseLayers2MathPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                           mlir::math::MathDialect, mlir::func::FuncDialect>();

    target.addLegalDialect<IE::IEDialect>();
    target.addIllegalOp<IE::TanhOp>();
    target.addIllegalOp<IE::CosOp>();
    target.addIllegalOp<IE::MaximumOp>();
    target.addIllegalOp<IE::MinimumOp>();
    target.addIllegalOp<IE::DivideOp>();
    target.addIllegalOp<IE::LogOp>();
    target.addIllegalOp<IE::ExpOp>();
    target.addIllegalOp<IE::BitwiseAndOp, IE::BitwiseOrOp, IE::BitwiseXorOp, IE::BitwiseNotOp>();
    target.addIllegalOp<IE::LogicalOrOp, IE::LogicalXorOp, IE::AndOp, IE::LogicalNotOp, IE::SelectOp>();
    target.addIllegalOp<IE::EqualOp, IE::NotEqualOp, IE::LessOp, IE::LessEqualOp, IE::GreaterOp, IE::GreaterEqualOp>();
    target.addIllegalOp<IE::SquaredDifferenceOp>();
    target.addIllegalOp<IE::SinOp>();
    target.addIllegalOp<IE::SqrtOp>();
    target.addIllegalOp<IE::ErfOp>();
    target.addIllegalOp<IE::RoundOp>();
    target.addIllegalOp<IE::ReLUOp, IE::LeakyReluOp, IE::ClampOp>();
    target.addIllegalOp<IE::AddOp, IE::MultiplyOp, IE::SubtractOp>();
    target.addIllegalOp<IE::ConvertOp>();
    target.addIllegalOp<IE::TanOp, IE::SinhOp, IE::CoshOp, IE::AtanOp, IE::AtanhOp>();
    target.addIllegalOp<IE::AbsOp>();
    target.addIllegalOp<IE::NegativeOp>();
    target.addIllegalOp<IE::SignOp>();
    target.addIllegalOp<IE::HSwishOp>();
    target.addIllegalOp<IE::HSigmoidOp>();

    auto populatePatterns = [&](mlir::RewritePatternSet& patternSet) {
        ConvertEltwiseLayersToMathPatterns::populateWithGenerated(patternSet);
        patternSet.add<IEEltwiseToLinalg<IE::MaximumOp>, IEEltwiseToLinalg<IE::MinimumOp>,
                       IEEltwiseToLinalg<IE::DivideOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::BitwiseAndOp>, IEEltwiseToLinalg<IE::BitwiseOrOp>,
                       IEEltwiseToLinalg<IE::BitwiseXorOp>, IEEltwiseToLinalg<IE::BitwiseNotOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::LogicalOrOp>, IEEltwiseToLinalg<IE::LogicalXorOp>,
                       IEEltwiseToLinalg<IE::AndOp>, IEEltwiseToLinalg<IE::LogicalNotOp>,
                       IEEltwiseToLinalg<IE::SelectOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::EqualOp>, IEEltwiseToLinalg<IE::NotEqualOp>, IEEltwiseToLinalg<IE::LessOp>,
                       IEEltwiseToLinalg<IE::LessEqualOp>, IEEltwiseToLinalg<IE::GreaterOp>,
                       IEEltwiseToLinalg<IE::GreaterEqualOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::SquaredDifferenceOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::ErfOp>, IEEltwiseToLinalg<IE::RoundOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::SinOp>, IEEltwiseToLinalg<IE::CosOp>>(&ctx);
        patternSet
                .add<IEEltwiseToLinalg<IE::ReLUOp>, IEEltwiseToLinalg<IE::LeakyReluOp>, IEEltwiseToLinalg<IE::ClampOp>>(
                        &ctx);
        patternSet.add<IEEltwiseToLinalg<IE::AddOp>, IEEltwiseToLinalg<IE::MultiplyOp>,
                       IEEltwiseToLinalg<IE::SubtractOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::ConvertOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::TanOp>, IEEltwiseToLinalg<IE::AtanhOp>, IEEltwiseToLinalg<IE::SinhOp>,
                       IEEltwiseToLinalg<IE::CoshOp>>(&ctx);
        patternSet.add<IEEltwiseToLinalg<IE::AbsOp>, IEEltwiseToLinalg<IE::NegativeOp>, IEEltwiseToLinalg<IE::SignOp>,
                       IEEltwiseToLinalg<IE::HSwishOp>, IEEltwiseToLinalg<IE::HSigmoidOp>>(&ctx);
    };

    // E#172607 [ShaveCodeGen] Make Linalg lowering pass run on CodeGenCapsuleOps
    func->walk([&](IE::CodeGenCapsuleOp capsuleOp) {
        mlir::RewritePatternSet patterns(&ctx);
        populatePatterns(patterns);
        if (mlir::failed(mlir::applyPartialConversion(capsuleOp, target, std::move(patterns)))) {
            signalPassFailure();
        }
    });
}

}  // namespace

//
// createConvertEltwiseLayers2MathPass
//

std::unique_ptr<mlir::Pass> ShaveCodeGen::createConvertEltwiseLayers2MathPass(Logger log) {
    return std::make_unique<ConvertEltwiseLayers2MathPass>(log);
}
