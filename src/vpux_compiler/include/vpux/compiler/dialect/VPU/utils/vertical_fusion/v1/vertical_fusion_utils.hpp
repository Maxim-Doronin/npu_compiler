//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v1 {
// check if whole operation is in CMX
bool isCmxOperation(mlir::Operation* operation, const bool checkTilingType);
}  // namespace vpux::VPU::VF::v1
