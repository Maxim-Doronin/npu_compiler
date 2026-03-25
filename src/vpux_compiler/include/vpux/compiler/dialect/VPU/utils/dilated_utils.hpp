//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>

namespace vpux::VPU {

/*
 * Check if the op is a SEP DWConv
 */
bool isSEPDWConv(mlir::Operation* op);

}  // namespace vpux::VPU
