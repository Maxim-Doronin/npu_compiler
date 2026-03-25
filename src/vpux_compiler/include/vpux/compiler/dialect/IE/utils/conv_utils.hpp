//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {

mlir::Value buildDwWeights(const mlir::Location& loc, int64_t OC, const mlir::Type& elementType,
                           mlir::PatternRewriter& rewriter);

}  // namespace vpux::IE
