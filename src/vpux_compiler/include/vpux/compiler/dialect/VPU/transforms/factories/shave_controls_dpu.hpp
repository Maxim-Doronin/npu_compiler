//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {
namespace VPU {

bool getShaveControlsDpu(config::ArchKind arch);
size_t getDpuDebugDataSize(config::ArchKind /*arch*/);
size_t getDPUInvariantDataSize(config::ArchKind arch);
size_t getDPUVariantDataSize(config::ArchKind arch);
}  // namespace VPU
}  // namespace vpux
