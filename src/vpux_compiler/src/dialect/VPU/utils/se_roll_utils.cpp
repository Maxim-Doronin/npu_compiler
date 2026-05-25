//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"

using namespace vpux;
using namespace VPU;

DimArr VPU::getRollSEPConvTilingOrder(VPU::SERollAttr seAttr) {
    const auto shift = parseIntArrayAttr<int64_t>(seAttr.getShift());
    if (shift[SE_ROLL_SPATIAL_H] != 0 && shift[SE_ROLL_SPATIAL_W] != 0) {
        return SmallVector<Dim>{Dims4D::Act::C};
    } else if (shift[SE_ROLL_SPATIAL_H] != 0) {
        return DimArr{Dims4D::Act::W, Dims4D::Act::C};
    } else {
        return DimArr{Dims4D::Act::H, Dims4D::Act::C};
    }
}

bool VPU::isRollSEPConvCompatibleWithClusterStrategy(VPU::SERollAttr seAttr, VPU::MultiClusterStrategy strategy) {
    const auto shift = parseIntArrayAttr<int64_t>(seAttr.getShift());
    if (shift[SE_ROLL_SPATIAL_H] != 0 && (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
                                          strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
                                          strategy == VPU::MultiClusterStrategy::SplitOverHeightKernel ||
                                          strategy == VPU::MultiClusterStrategy::SplitOverHeightWidth ||
                                          strategy == VPU::MultiClusterStrategy::HKSwitch)) {
        return false;
    }
    if (shift[SE_ROLL_SPATIAL_W] != 0 && (strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
                                          strategy == VPU::MultiClusterStrategy::SplitOverHeightWidth)) {
        return false;
    }

    if (shift[SE_ROLL_SPATIAL_H] != 0 && shift[SE_ROLL_SPATIAL_W] != 0) {
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverKernel;
    }
    return true;
}
