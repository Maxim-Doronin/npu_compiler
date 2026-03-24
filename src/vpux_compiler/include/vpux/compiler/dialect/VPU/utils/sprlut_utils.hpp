//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux::VPU {
class PPEAttr;
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {

constexpr auto SPRLUT_ALIGNMENT_REQUIREMENT = 32;

bool hasSprLUTAttribute(PPEAttr ppeAttr);

Byte getSprLUTSize(PPEAttr ppeAttr);

void addSprLutBufferIfPresent(PPEAttr ppeAttr, SmallVector<Byte>& buffers);

}  // namespace VPU
}  // namespace vpux
