//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/Types.h>

namespace vpux {
namespace VPUIP {

mlir::Type setCompressionState(mlir::Type type, VPUIP::CompressionState compression);

VPUIP::CompressionStateAttr getCompressionStateAttr(mlir::Type type);
VPUIP::CompressionState getCompressionState(mlir::Type type);

bool isSupportedBufferSizeForCompression(vpux::NDTypeInterface ndType);

}  // namespace VPUIP
}  // namespace vpux
