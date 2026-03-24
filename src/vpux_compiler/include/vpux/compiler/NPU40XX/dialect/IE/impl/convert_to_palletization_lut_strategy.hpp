//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::IE::arch40xx {

class ConvertToPalletizationLUTStrategy : public IConversionPassStrategy {
public:
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;
    void markOpLegality(mlir::ConversionTarget& target, Logger& log) const override final;
};

}  // namespace vpux::IE::arch40xx
