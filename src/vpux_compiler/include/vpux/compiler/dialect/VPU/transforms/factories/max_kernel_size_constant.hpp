//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/max_kernel_size_constant.hpp"

#include <cstdint>

namespace vpux::config {
enum class ArchKind : uint64_t;
}

namespace vpux {
namespace VPU {

VPU::MaxKernelSizeConstant getMaxKernelSizeConstant(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
