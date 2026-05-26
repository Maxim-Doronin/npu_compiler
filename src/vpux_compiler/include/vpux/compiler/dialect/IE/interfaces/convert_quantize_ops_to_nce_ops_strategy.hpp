//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {

class IConvertQuantizeOpsToNceOpsStrategy {
public:
    virtual void prepareAvgPool(mlir::RewritePatternSet& toAvgPoolPatterns, mlir::MLIRContext& ctx,
                                Logger& log) const = 0;
    virtual void prepareEltwise(mlir::RewritePatternSet& toEltwisePatterns, mlir::MLIRContext& ctx,
                                Logger& log) const = 0;
    virtual void prepareQuantToConv(mlir::RewritePatternSet& quantToConvPatterns, mlir::MLIRContext& ctx,
                                    Logger& log) const = 0;

    virtual ~IConvertQuantizeOpsToNceOpsStrategy() = default;
};

}  // namespace vpux::IE
