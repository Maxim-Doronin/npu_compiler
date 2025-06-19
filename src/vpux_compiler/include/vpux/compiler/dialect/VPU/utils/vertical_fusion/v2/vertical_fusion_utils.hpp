//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v2 {

// check if whole operation is in CMX
bool isCmxOperation(mlir::Operation* operation, const bool checkTilingType);

// check if previous operation has some DDR users apart from VF
bool hasBeforeDDRUsers(mlir::Operation* prevOp, mlir::Operation* nextOp);

// Check if the op has multi view op user with shape changed, which will cause the output to be spilled
bool hasOutputSpilledForDifferentDataSizeUses(mlir::Operation* op);

// Check if the op's output is tiled on same axis as the disributed output type's tiling axis
bool outputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op);

// Check if the op's input is tiled on same axis as the disributed input type's tiling axis
bool inputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op, mlir::Value operand);

}  // namespace vpux::VPU::VF::v2
