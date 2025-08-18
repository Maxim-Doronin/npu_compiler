//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"

namespace vpux::VPUIP {
class SwKernelOp;
}

namespace vpux::VPU {

OutputTiling DetectionOutputSortOpOutputTiling(const vpux::TileInfo& firstOutputTile);
InputTiling DetectionOutputSortOpInputTiling(const vpux::TileInfo& firstOutputTile, int numShaves);
InputTiling DetectionOutputSortOpInputTilingOnShave(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                    int tileId, int tileCount, Logger log);

OutputTiling GRUSequenceOutputTiling(const vpux::TileInfo& firstOutputTile);
OutputTiling lstmSequenceOutputTiling(const vpux::TileInfo& firstOutputTile);
OutputTiling DynamicQuantizeOutputTiling(const vpux::TileInfo& firstOutputTile);

}  // namespace vpux::VPU
