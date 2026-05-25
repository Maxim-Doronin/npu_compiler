//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::IE::arch50xx {

class FuseQuantizedOpsStrategy : public IGreedilyPassStrategy {
public:
    FuseQuantizedOpsStrategy(const bool seOpsEnabled): _seOpsEnabled(seOpsEnabled) {
    }

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;

private:
    bool _seOpsEnabled;
};

}  // namespace vpux::IE::arch50xx
