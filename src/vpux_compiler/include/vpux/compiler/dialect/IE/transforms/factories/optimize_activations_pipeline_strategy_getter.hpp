//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_strategies.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE {

//
// OptimizeActivationsPipelineStrategy
//

class OptimizeActivationsPipelineStrategy final : public IDynamicRewriterStrategy {
public:
    explicit OptimizeActivationsPipelineStrategy(bool enableSEOps, bool enableFuseClamp)
            : _enableSEOps(enableSEOps), _enableFuseClamp(enableFuseClamp) {
    }

    void registerRewriters(RewriterRegistry& registry, Logger& log) const override;

private:
    bool _enableSEOps = false;
    bool _enableFuseClamp = false;
};

std::unique_ptr<IDynamicRewriterStrategy> createOptimizeActivationsPipelineStrategy(mlir::func::FuncOp funcOp,
                                                                                    bool enableSEOps = false,
                                                                                    bool enableFuseClamp = false);

}  // namespace vpux::IE
