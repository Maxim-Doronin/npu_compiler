//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

constexpr StringRef DYNAMIC_DIM_ALIGNMENT = "DynamicDimAlignment";
bool hasDynamicDimAlignment(mlir::Operation* op);
void setDynamicDimAlignment(mlir::Operation* op);
void removeDynamicDimAlignment(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
