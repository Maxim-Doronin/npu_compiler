//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::VPUIPDPU::arch40xx::General {

mlir::LogicalResult buildGeneralForceInvRead(mlir::OpBuilder& builder, const mlir::Location& loc,
                                             std::optional<bool> forceInvRead);

}  // namespace vpux::VPUIPDPU::arch40xx::General
