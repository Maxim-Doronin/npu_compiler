//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPUIP {

//! @brief Allocates buffers for the results of the funcOps.
//! Only pass in the funcOp and callOp that need buffer allocation.
template <typename CopyOp = VPUIP::CopyOp>
void allocateBuffersForNetResults(const mlir::DenseSet<mlir::CallOpInterface>& callOps,
                                  const mlir::DenseSet<mlir::func::FuncOp>& funcOps, Logger& log);

}  // namespace vpux::VPUIP
