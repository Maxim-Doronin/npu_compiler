//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

namespace vpux::VPU {

DimArr getRollSEPConvTilingOrder(VPU::SERollAttr seAttr);
bool isRollSEPConvCompatibleWithClusterStrategy(VPU::SERollAttr seAttr, VPU::MultiClusterStrategy strategy);

}  // namespace vpux::VPU
