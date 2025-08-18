//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::arch40xx {

class UnrollSingleClusterPermuteDMA {
public:
    static mlir::LogicalResult unroll(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter, int64_t portCount,
                                      Logger logger);
};

mlir::LogicalResult rewritePermuteDMA(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter, int64_t portCount,
                                      Logger logger);

}  // namespace vpux::arch40xx
