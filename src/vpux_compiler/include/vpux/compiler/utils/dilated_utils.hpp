//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {

NDTypeInterface getDilatedType(vpux::NDTypeInterface origType, ShapeRef dilations);

/*
 * Check if the op is a SEP DWConv
 */
bool isSEPDWConv(mlir::Operation* op);

}  // namespace vpux
