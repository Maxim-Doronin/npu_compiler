//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace vpux::config {
enum class ArchKind : uint64_t;
}  // namespace vpux::config

namespace vpux::VPU {
enum class TaskType : uint64_t;
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {

uint32_t getDefaultTaskListCount(VPU::TaskType taskType, config::ArchKind archKind);

}  // namespace VPU
}  // namespace vpux
