//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// ============================================================================
// TEMPORARY WORKAROUND IMPLEMENTATION
//
// These files (Approximation.hpp and Approximation.cpp)
// serve as temporary storage for various functions and utilities required
// for our expansion work. Due to our current integration with an older version
// of LLVM which does not expose some of the new APIs (such as SinAndCosApproximation)
// via official headers, we have copied the necessary functionality into these files.
//
// In addition to the SinAndCosApproximation functionality, we plan to add and
// extend other functions here as needed during this transitional phase.
//
// NOTE:
//   - This is a temporary solution until we update LLVM and gain direct access to
//     the official implementations via provided headers.
//   - The code in these files is subject to change and will eventually be removed
//     or refactored once the LLVM update is complete.
//   - Please ensure that any new functions added to these files are clearly marked
//     as part of this temporary workaround, and are reviewed during the migration.
// ============================================================================

#include "vpux/compiler/conversion/passes/ShaveCodeGen/Approximation.hpp"
#include <mlir/Dialect/Math/Transforms/Passes.h>
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace mlir;
using namespace mlir::math;
using namespace mlir::vector;

namespace vpux::ShaveCodeGen {

// Returns vector shape if the type is a vector. Returns an empty shape if it is
// not a vector.
static inline ArrayRef<int64_t> vectorShape(Type type) {
    auto vectorType = dyn_cast<VectorType>(type);
    return vectorType ? vectorType.getShape() : ArrayRef<int64_t>();
}

static inline ArrayRef<int64_t> vectorShape(Value value) {
    return vectorShape(value.getType());
}

//----------------------------------------------------------------------------//
// Helper functions to create constants.
//----------------------------------------------------------------------------//

static inline Value f32Cst(ImplicitLocOpBuilder& builder, double value) {
    return builder.create<arith::ConstantOp>(builder.getF32FloatAttr(value));
}

static inline Value i32Cst(ImplicitLocOpBuilder& builder, int32_t value) {
    return builder.create<arith::ConstantOp>(builder.getI32IntegerAttr(value));
}

//----------------------------------------------------------------------------//
// Broadcast scalar types and values into vector types and values.
//----------------------------------------------------------------------------//

// Broadcasts scalar type into vector type (iff shape is non-scalar).
static inline Type broadcast(Type type, ArrayRef<int64_t> shape) {
    assert(!isa<VectorType>(type) && "must be scalar type");
    return !shape.empty() ? VectorType::get(shape, type) : type;
}

// Broadcasts scalar value into vector (iff shape is non-scalar).
static inline Value broadcast(ImplicitLocOpBuilder& builder, Value value, ArrayRef<int64_t> shape) {
    assert(!isa<VectorType>(value.getType()) && "must be scalar value");
    auto type = broadcast(value.getType(), shape);
    return !shape.empty() ? builder.create<BroadcastOp>(type, value) : value;
}
//----------------------------------------------------------------------------//
// Sin and Cos approximation.
//----------------------------------------------------------------------------//

#define TWO_OVER_PI 0.6366197723675813430755350534900574481378385829618257949906693762L
#define PI_OVER_2 1.5707963267948966192313216916397514420985846996875529104874722961L

// Approximates sin(x) or cos(x) by finding the best approximation polynomial in
// the reduced range [0, pi/2] for both sin(x) and cos(x). Then given y in the
// reduced range sin(x) will be computed as sin(y), -sin(y), cos(y) or -cos(y).
template <bool isSine, typename OpTy>
mlir::LogicalResult SinAndCosApproximation<isSine, OpTy>::matchAndRewrite(OpTy op,
                                                                          mlir::PatternRewriter& rewriter) const {
    using namespace mlir;
    using namespace mlir::math;

    static_assert(llvm::is_one_of<OpTy, math::SinOp, math::CosOp>::value,
                  "SinAndCosApproximation pattern expects math::SinOp or math::CosOp");

    if (!getElementTypeOrSelf(op.getOperand()).isF32()) {
        return rewriter.notifyMatchFailure(op, "unsupported operand type");
    }

    ArrayRef<int64_t> shape = vectorShape(op.getOperand());

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto bcast = [&](Value value) -> Value {
        return broadcast(builder, value, shape);
    };
    auto mul = [&](Value a, Value b) -> Value {
        return builder.create<arith::MulFOp>(a, b);
    };
    auto sub = [&](Value a, Value b) -> Value {
        return builder.create<arith::SubFOp>(a, b);
    };
    auto floor = [&](Value a) {
        return builder.create<math::FloorOp>(a);
    };

    auto i32Vec = broadcast(builder.getI32Type(), shape);
    auto fPToSingedInteger = [&](Value a) -> Value {
        return builder.create<arith::FPToSIOp>(i32Vec, a);
    };

    auto modulo4 = [&](Value a) -> Value {
        return builder.create<arith::AndIOp>(a, bcast(i32Cst(builder, 3)));
    };

    auto isEqualTo = [&](Value a, Value b) -> Value {
        return builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, a, b);
    };

    auto isGreaterThan = [&](Value a, Value b) -> Value {
        return builder.create<arith::CmpIOp>(arith::CmpIPredicate::sgt, a, b);
    };

    auto select = [&](Value cond, Value t, Value f) -> Value {
        return builder.create<arith::SelectOp>(cond, t, f);
    };

    auto fmla = [&](Value a, Value b, Value c) {
        return builder.create<math::FmaOp>(a, b, c);
    };

    auto bitwiseOr = [&](Value a, Value b) {
        return builder.create<arith::OrIOp>(a, b);
    };

    Value twoOverPi = bcast(f32Cst(builder, (float)TWO_OVER_PI));
    Value piOverTwo = bcast(f32Cst(builder, (float)PI_OVER_2));

    Value x = op.getOperand();

    Value k = floor(mul(x, twoOverPi));

    Value y = sub(x, mul(k, piOverTwo));

    Value cstOne = bcast(f32Cst(builder, 1.0));
    Value cstNegativeOne = bcast(f32Cst(builder, -1.0));

    Value cstSC2 = bcast(f32Cst(builder, -0.16666667163372039794921875f));
    Value cstSC4 = bcast(f32Cst(builder, 8.333347737789154052734375e-3f));
    Value cstSC6 = bcast(f32Cst(builder, -1.9842604524455964565277099609375e-4f));
    Value cstSC8 = bcast(f32Cst(builder, 2.760012648650445044040679931640625e-6f));
    Value cstSC10 = bcast(f32Cst(builder, -2.50293279435709337121807038784027099609375e-8f));

    Value cstCC2 = bcast(f32Cst(builder, -0.5f));
    Value cstCC4 = bcast(f32Cst(builder, 4.166664183139801025390625e-2f));
    Value cstCC6 = bcast(f32Cst(builder, -1.388833043165504932403564453125e-3f));
    Value cstCC8 = bcast(f32Cst(builder, 2.47562347794882953166961669921875e-5f));
    Value cstCC10 = bcast(f32Cst(builder, -2.59630184018533327616751194000244140625e-7f));

    Value kMod4 = modulo4(fPToSingedInteger(k));

    Value kR0 = isEqualTo(kMod4, bcast(i32Cst(builder, 0)));
    Value kR1 = isEqualTo(kMod4, bcast(i32Cst(builder, 1)));
    Value kR2 = isEqualTo(kMod4, bcast(i32Cst(builder, 2)));
    Value kR3 = isEqualTo(kMod4, bcast(i32Cst(builder, 3)));

    Value sinuseCos = isSine ? bitwiseOr(kR1, kR3) : bitwiseOr(kR0, kR2);
    Value negativeRange = isSine ? isGreaterThan(kMod4, bcast(i32Cst(builder, 1))) : bitwiseOr(kR1, kR2);

    Value y2 = mul(y, y);

    Value base = select(sinuseCos, cstOne, y);
    Value cstC2 = select(sinuseCos, cstCC2, cstSC2);
    Value cstC4 = select(sinuseCos, cstCC4, cstSC4);
    Value cstC6 = select(sinuseCos, cstCC6, cstSC6);
    Value cstC8 = select(sinuseCos, cstCC8, cstSC8);
    Value cstC10 = select(sinuseCos, cstCC10, cstSC10);

    Value v1 = fmla(y2, cstC10, cstC8);
    Value v2 = fmla(y2, v1, cstC6);
    Value v3 = fmla(y2, v2, cstC4);
    Value v4 = fmla(y2, v3, cstC2);
    Value v5 = fmla(y2, v4, cstOne);
    Value v6 = mul(base, v5);

    Value approximation = select(negativeRange, mul(cstNegativeOne, v6), v6);

    rewriter.replaceOp(op, approximation);

    return success();
}
template struct SinAndCosApproximation<true, mlir::math::SinOp>;
template struct SinAndCosApproximation<false, mlir::math::CosOp>;

}  // namespace vpux::ShaveCodeGen
