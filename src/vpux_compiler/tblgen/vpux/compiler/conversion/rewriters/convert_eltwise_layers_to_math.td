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

//
// IE.Cos -> math.Cos
//

def createCosOp :
        NativeCodeCall<[{
            $_builder.create<mlir::math::CosOp>($_loc, $0)
        }]>;

def RewriteCos :
        Pat<
            (IE_CosOp $input),
            (createCosOp $input)
        >;

def createFastMathAttr :
        NativeCodeCall<[{
            mlir::arith::FastMathFlagsAttr::get(
                       $_builder.getContext(),
                       static_cast<mlir::arith::FastMathFlags>($0.getValue().getLimitedValue()))
        }]>;

//
// IE.Log -> math.Log
//

def RewriteLog :
        Pat<
            (IE_LogOp $input),
            (Math_LogOp $input, (createFastMathAttr FASTMATH_APPROX_FUNC))
        >;


//
// IE.Exp -> math.Exp
//

def RewriteExp :
        Pat<
            (IE_ExpOp $input),
            (Math_ExpOp $input, (createFastMathAttr FASTMATH_APPROX_FUNC))
        >;

#endif
