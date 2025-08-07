//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace VPU {
//
// ITilingAlgorithm
//

// The interface of tiling algorithm
class ITilingAlgorithm {
public:
    virtual ~ITilingAlgorithm() = default;

    virtual mlir::LogicalResult applyTiling(mlir::Operation* operation, mlir::RewriterBase& builder, Logger log) = 0;
};
}  // namespace VPU
}  // namespace vpux
