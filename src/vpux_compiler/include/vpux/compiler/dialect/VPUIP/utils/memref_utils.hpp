//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>

namespace vpux {
class NDTypeInterface;
}

namespace vpux::VPUIP {

mlir::IntegerAttr getAllocSizeAttr(mlir::Type type);
vpux::NDTypeInterface setAllocSizeAttr(vpux::NDTypeInterface type, int64_t allocSize);

}  // namespace vpux::VPUIP
