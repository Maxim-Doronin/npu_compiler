//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"

namespace vpux::VPUIP {
bool areStridesCompatible(const MemStrides& inStrides, Bit inElemSize, const MemStrides& outStrides, Bit outElemSize);
}
