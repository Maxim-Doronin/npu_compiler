//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/strides_utils.hpp"

namespace vpux {
namespace VPUIP {

std::optional<MemDimArr> deduceLegalOutputMemDims(MemShapeRef inMemShape, MemShapeRef outMemShape, MemDim inMemDim);
mlir::FailureOr<vpux::NDTypeInterface> updateStridesForReshape(const vpux::NDTypeInterface& inType,
                                                               const vpux::NDTypeInterface& outType);
bool isInAndOutStridesCompatible(const vpux::NDTypeInterface& inType, const vpux::NDTypeInterface& outType);

}  // namespace VPUIP
}  // namespace vpux
