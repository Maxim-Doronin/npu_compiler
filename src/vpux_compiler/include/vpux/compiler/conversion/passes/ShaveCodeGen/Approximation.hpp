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

#ifndef SHAVECODEGEN_MATH_TRANSFORMS_APPROXIMATION_H
#define SHAVECODEGEN_MATH_TRANSFORMS_APPROXIMATION_H

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"

namespace vpux::ShaveCodeGen {

template <bool isSine, typename OpTy>
struct SinAndCosApproximation : public mlir::OpRewritePattern<OpTy> {
public:
    using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(OpTy op, mlir::PatternRewriter& rewriter) const override;
};
}  // namespace vpux::ShaveCodeGen

#endif  // SHAVECODEGEN_MATH_TRANSFORMS_APPROXIMATION_H
