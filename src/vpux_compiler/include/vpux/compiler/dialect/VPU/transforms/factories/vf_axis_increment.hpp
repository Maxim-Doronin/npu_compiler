//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_axis_increment.hpp"

namespace vpux::VPU {

/*
   Find appropriate class to calculate changes for each axis for VF
*/
std::unique_ptr<IVFAxisIncrement> getVFAxisIncrement(Dim axis);

}  // namespace vpux::VPU
