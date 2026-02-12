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
// BuildMemPermuteProcessingStrategy
//

class MemPermuteProcessingPipelineStrategy final : public IDynamicRewriterStrategy {
public:
    explicit MemPermuteProcessingPipelineStrategy(bool seOpsEnabled, bool enableAdjustConvShapePass)
            : _seOpsEnabled(seOpsEnabled), _enableAdjustConvShapePass(enableAdjustConvShapePass) {
    }

    void registerRewriters(RewriterRegistry& registry, Logger& log) const override;

private:
    bool _seOpsEnabled;
    bool _enableAdjustConvShapePass;
};

std::unique_ptr<IDynamicRewriterStrategy> createMemPermuteProcessingPipelineStrategy(mlir::func::FuncOp funcOp,
                                                                                     bool seOpsEnabled,
                                                                                     bool enableAdjustConvShapePass);

}  // namespace vpux::IE
