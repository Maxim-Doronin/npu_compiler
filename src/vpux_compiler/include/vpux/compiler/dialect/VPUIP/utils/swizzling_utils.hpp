//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include <cstdint>

namespace vpux {
class NDTypeInterface;
}  // namespace vpux

namespace vpux::VPUIP {
class SwizzlingSchemeAttr;
class DistributedBufferType;
}  // namespace vpux::VPUIP

namespace vpux::config {
enum class ArchKind : uint64_t;
}  // namespace vpux::config

namespace vpux::VPUIP {

VPUIP::SwizzlingSchemeAttr createSwizzlingSchemeAttr(mlir::MLIRContext* ctx, config::ArchKind archKind,
                                                     int64_t swizzlingKey);

VPUIP::SwizzlingSchemeAttr getSwizzlingSchemeAttr(mlir::Type type);

// Retrieve swizzling key setting embedded in layout with buffer types
int64_t getSwizzlingKey(mlir::Type type);

mlir::Type setSwizzlingKey(mlir::Type type, int64_t swizzlingKey, config::ArchKind archKind);

vpux::NDTypeInterface updateSwizzlingSchemeBasedOnDistributedType(VPUIP::DistributedBufferType inputType,
                                                                  vpux::NDTypeInterface newType);

}  // namespace vpux::VPUIP
