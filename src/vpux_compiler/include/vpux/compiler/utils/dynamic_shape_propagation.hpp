//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Support/LLVM.h>
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {

void assignDynamicTypeComponents(TypeComponents& typeComponents, VPU::BoundsRepresentation boundsRepresentation,
                                 ArrayRef<int64_t> shape, ArrayRef<int64_t> bounds);

}  // namespace vpux
