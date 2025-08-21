//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Support/LLVM.h>

namespace vpux::VPU {
enum class BoundsRepresentation : uint64_t;
}  // namespace vpux::VPU

namespace vpux {

void assignDynamicTypeComponents(TypeComponents& typeComponents, VPU::BoundsRepresentation boundsRepresentation,
                                 ArrayRef<int64_t> shape, ArrayRef<int64_t> bounds);

}  // namespace vpux
