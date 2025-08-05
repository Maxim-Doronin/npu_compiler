//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {

class IConvertQuantizeOpsToNceOpsStrategy {
public:
    virtual void prepareAvgPool(mlir::ConversionTarget& toAvgPoolTarget, mlir::RewritePatternSet& toAvgPoolPatterns,
                                mlir::MLIRContext& ctx, Logger& log) const = 0;
    virtual void prepareEltwise(mlir::ConversionTarget& toEltwiseTarget, mlir::RewritePatternSet& toEltwisePatterns,
                                mlir::MLIRContext& ctx, Logger& log) const = 0;
    virtual void prepareQuantToConv(mlir::ConversionTarget& quantToConvTarget,
                                    mlir::RewritePatternSet& quantToConvPatterns, mlir::MLIRContext& ctx,
                                    Logger& log) const = 0;

    virtual ~IConvertQuantizeOpsToNceOpsStrategy() = default;
};

}  // namespace vpux::IE
