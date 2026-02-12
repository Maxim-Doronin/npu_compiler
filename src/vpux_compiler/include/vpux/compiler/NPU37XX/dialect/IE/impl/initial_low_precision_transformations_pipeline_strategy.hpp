//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_strategies.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE::arch37xx {

//
// InitialLowPrecisionTransformationsPipelineStrategy
//

class InitialLowPrecisionTransformationsPipelineStrategy final : public IDynamicRewriterStrategy {
public:
    explicit InitialLowPrecisionTransformationsPipelineStrategy(mlir::func::FuncOp funcOp): _funcOp(funcOp) {
    }

    void registerRewriters(RewriterRegistry& registry, Logger& log) const override;

private:
    mlir::func::FuncOp _funcOp;
};

}  // namespace vpux::IE::arch37xx
