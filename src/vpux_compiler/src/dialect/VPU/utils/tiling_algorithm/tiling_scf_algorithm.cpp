//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_scf_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"

using namespace vpux;
using namespace VPU;

mlir::LogicalResult TilingSCFAlgorithm::applyTiling(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                    Logger /*log*/) {
    return VPU::applySCFTiling(operation, builder);
}
