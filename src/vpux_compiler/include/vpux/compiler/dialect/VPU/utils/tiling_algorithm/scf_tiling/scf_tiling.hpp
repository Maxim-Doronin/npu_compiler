//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::VPU {

// Apply tiling using SCF dialect
mlir::LogicalResult applySCFTiling(mlir::Operation* operation, mlir::RewriterBase& builder);

SmallVector<mlir::OpFoldResult> staticTileSizeComputation(mlir::OpBuilder& builder, mlir::Operation* operation,
                                                          ShapeRef strategy, ShapeRef outputShape);

SmallVector<mlir::OpFoldResult> dynamicTileSizeComputation(mlir::OpBuilder& builder, mlir::Operation* operation,
                                                           ShapeRef strategy);

}  // namespace vpux::VPU
