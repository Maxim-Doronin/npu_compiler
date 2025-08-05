//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {
namespace VPU {

bool getShaveControlsDpu(VPU::ArchKind arch);
size_t getDpuDebugDataSize(VPU::ArchKind /*arch*/);
size_t getDPUInvariantDataSize(VPU::ArchKind arch);
size_t getDPUVariantDataSize(VPU::ArchKind arch);
}  // namespace VPU
}  // namespace vpux
