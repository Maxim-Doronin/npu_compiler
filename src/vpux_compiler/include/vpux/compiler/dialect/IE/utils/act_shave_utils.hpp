//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>

namespace vpux {
namespace IE {

bool isActShaveKernel(mlir::Operation* operation);

}  // namespace IE
}  // namespace vpux
