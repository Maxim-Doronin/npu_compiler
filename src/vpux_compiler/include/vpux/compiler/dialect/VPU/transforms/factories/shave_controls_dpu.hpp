//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace vpux::config {
enum class ArchKind : uint64_t;
}

namespace vpux {
namespace VPU {

bool getShaveControlsDpu(config::ArchKind arch);
size_t getDpuDebugDataSize(config::ArchKind /*arch*/);
size_t getDPUInvariantDataSize(config::ArchKind arch);
size_t getDPUVariantDataSize(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
