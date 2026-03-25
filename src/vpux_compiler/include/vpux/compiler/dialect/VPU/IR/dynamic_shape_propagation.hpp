//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"

#include <cstdint>

namespace vpux {
struct TypeComponents;
}  // namespace vpux

namespace vpux::VPU {
enum class BoundsRepresentation : uint64_t;

void assignDynamicTypeComponents(vpux::TypeComponents& typeComponents, VPU::BoundsRepresentation boundsRepresentation,
                                 ArrayRef<int64_t> shape, ArrayRef<int64_t> bounds);
}  // namespace vpux::VPU
