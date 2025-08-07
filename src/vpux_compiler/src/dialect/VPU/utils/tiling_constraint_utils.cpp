//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_constraint_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

constexpr double NPU37XX_FRAGMENTATION_AVOID_RATIO_PIPELINING_LARGE_WEIGHTS = 0.45;

const std::unordered_map<VPU::ArchKind, double> fragmentationDefaultRatioForPipelinlingMap = {
        {VPU::ArchKind::NPU37XX, NPU37XX_FRAGMENTATION_AVOID_RATIO_PIPELINING_LARGE_WEIGHTS},
        {VPU::ArchKind::NPU40XX, NPU37XX_FRAGMENTATION_AVOID_RATIO_PIPELINING_LARGE_WEIGHTS},
};

double VPU::getFragmentationAvoidRatioPipeliningLargeWeights(VPU::ArchKind archKind) {
    auto iter = fragmentationDefaultRatioForPipelinlingMap.find(archKind);
    VPUX_THROW_WHEN(iter == fragmentationDefaultRatioForPipelinlingMap.end(),
                    "getFragmentationAvoidRatioPipeliningLargeWeights: Unsupported arch {0}", archKind);
    return iter->second;
}
