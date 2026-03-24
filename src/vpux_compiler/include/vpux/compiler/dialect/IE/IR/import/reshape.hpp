//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>

namespace vpux::IE::Reshape {

/** @brief Returns an output shape of an IE::ReshapeOp.
 */
llvm::FailureOr<llvm::SmallVector<int64_t>> parseOutShape(mlir::Location loc, mlir::ValueRange opInputs,
                                                          bool specialZero);

}  // namespace vpux::IE::Reshape
