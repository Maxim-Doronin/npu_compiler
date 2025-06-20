//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef FRAGMENTATION_AVOID_RATIO_PIPELINING_LARGE_WEIGHTS =
        "VPU.FragmentationAvoidRatioPipeliningLargeWeights";

double getFragmentationAvoidRatioPipeliningLargeWeights(VPU::ArchKind archKind);
}  // namespace VPU
}  // namespace vpux
