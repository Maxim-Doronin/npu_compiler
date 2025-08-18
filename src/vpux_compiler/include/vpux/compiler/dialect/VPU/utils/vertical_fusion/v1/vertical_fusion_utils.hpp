//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_axis_increment.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v1 {
// check if whole operation is in CMX
bool isCmxOperation(mlir::Operation* operation, const bool checkTilingType);

// get the maximal valid tiling strategy for VF block between the given range of tiling strategy
mlir::FailureOr<SmallVector<int64_t>> getMaximalValidTilingStrategyFromRange(
        VerticalFusionOp subgraph, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log);

// get the minimal valid tiling strategy for VF block between the given range of tiling strategy
mlir::FailureOr<SmallVector<int64_t>> getMinimalValidTilingStrategyFromRange(
        VerticalFusionOp subgraph, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log);

// if the maxTile is too large, return the cbrt of it if it's a valid max tile candidate
std::optional<int64_t> getCbrtMaxTileCandidate(int64_t minTile, int64_t maxTile,
                                               std::unique_ptr<IVFAxisIncrement>& axisIncrement);
}  // namespace vpux::VPU::VF::v1
