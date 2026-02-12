//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

/*
   Class for DepthSpace to TransposedConv conversion verifier
*/
class D2SToTransposedConvVerifierBase {
public:
    virtual ~D2SToTransposedConvVerifierBase() = default;

    virtual mlir::LogicalResult isBeneficialConversion(Logger log, mlir::PatternRewriter& rewriter,
                                                       IE::DepthToSpaceOp d2sOp, const bool seOpsEnabled) const = 0;
};

}  // namespace IE
}  // namespace vpux
