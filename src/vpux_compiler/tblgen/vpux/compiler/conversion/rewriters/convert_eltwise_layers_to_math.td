//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_ELTWISE_LAYERS_TO_MATH
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_ELTWISE_LAYERS_TO_MATH

include "vpux/compiler/dialect/IE/ops.td"
include "mlir/Dialect/Math/IR/MathOps.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

// The following utilities convert tensors to and from types that can be
// directly used as linalg ops. The conversions keep the same memory
// representation, so they will end up being no-ops.

def toLinalgLayout :
        NativeCodeCall<[{
           vpux::ShaveCodeGen::convertToLinalgValue($0, $_builder)
        }]>;

def fromLinalgLayout :
        NativeCodeCall<[{
           vpux::ShaveCodeGen::convertFromLinalgValue($0, $1.getType(), $_builder)
        }]>;

// Utilities

def createFastMathAttr :
        NativeCodeCall<[{
            mlir::arith::FastMathFlagsAttr::get(
                       $_builder.getContext(),
                       static_cast<mlir::arith::FastMathFlags>($0.getValue().getLimitedValue()))
        }]>;

// Conversion utility for unary operations.
multiclass SCGUnary<Op in_op, Op out_op, Attr fmf> {
   def _pat : Pat<(in_op $input),
                  (fromLinalgLayout
                      (out_op (toLinalgLayout $input),
                              (createFastMathAttr fmf)),
                      $input)>;
}

// Conversion patterns

defm LogPat : SCGUnary<IE_LogOp, Math_LogOp, FASTMATH_APPROX_FUNC>;
defm ExpPat : SCGUnary<IE_ExpOp, Math_ExpOp, FASTMATH_APPROX_FUNC>;
defm SqrtPat : SCGUnary<IE_SqrtOp, Math_SqrtOp, FASTMATH_APPROX_FUNC>;

#endif
