//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_strategies.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE {

//
// BuildBatchOpProcessingStrategy
//

class BatchOpProcessingPipelineStrategy final : public IDynamicRewriterStrategy {
public:
    explicit BatchOpProcessingPipelineStrategy(bool enableGroupedMatMul): _enableGroupedMatMul(enableGroupedMatMul) {
    }

    void registerRewriters(RewriterRegistry& registry, Logger& log) const override;

private:
    bool _enableGroupedMatMul;
};

std::unique_ptr<IDynamicRewriterStrategy> createBatchOpProcessingPipelineStrategy(mlir::func::FuncOp funcOp,
                                                                                  bool enableGroupedMatMul);

}  // namespace vpux::IE
