//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/Operation.h>

namespace vpux::VPU {
enum class MultiClusterStrategy : uint64_t;
}

namespace vpux::VPU {

bool checkStrategyCompatibilityReduce(VPU::MultiClusterStrategy strategy, size_t numTiles, ShapeRef inShape,
                                      ArrayRef<int64_t> axesVec);

bool fitIntoCMXReduce(mlir::Operation* operation, llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

bool fitIntoCMXReduce(mlir::Operation* operation, llvm::ArrayRef<vpux::NDTypeInterface> buffers);

}  // namespace vpux::VPU
