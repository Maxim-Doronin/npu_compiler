//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {
class NDTypeInterface;

NDTypeInterface getDilatedType(vpux::NDTypeInterface origType, ShapeRef dilations);

}  // namespace vpux
