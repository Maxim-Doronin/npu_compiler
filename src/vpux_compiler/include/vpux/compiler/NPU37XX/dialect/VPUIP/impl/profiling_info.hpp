//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIP::arch37xx {

mlir::Type getTimestampType(mlir::MLIRContext* ctx);
void setWorkloadIds(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

}  // namespace vpux::VPUIP::arch37xx
