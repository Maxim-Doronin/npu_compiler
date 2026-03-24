//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::ProfilingSectionOp::verify() {
    if (!mlir::isa_and_nonnull<net::DataInfoOp>(getOperation()->getParentOp())) {
        return errorAt(getOperation(), "Parent should be net::DataInfoOp");
    }
    return mlir::success();
}
