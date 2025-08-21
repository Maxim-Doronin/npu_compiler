//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attr_interfaces.hpp"
#include "vpux/compiler/dialect/core/interfaces/attr_interfaces.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::VPU {
class ExecutorKindAttr;
}  // namespace vpux::VPU

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/attributes.hpp.inc>

namespace vpux {
namespace VPUIP {

//
// SparsityCompressionAttr
//

VPUIP::SparsityCompressionAttr getSparsityCompressionAttr(mlir::Type type);
mlir::Type setSparsityCompressionAttr(mlir::Type type, VPUIP::SparsityCompressionAttr sparsityCompressionAttr);

VPUIP::SparsityCompressionAttr tileSparsityCompression(VPUIP::SparsityCompressionAttr sparsityCompression,
                                                       ShapeRef tileOffsets, ShapeRef tileShape);
mlir::Type tileTypeSparsityCompression(mlir::Type type, ShapeRef tileOffsets, ShapeRef tileShape);

}  // namespace VPUIP
}  // namespace vpux
