//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>

namespace vpux {
namespace IE {
/// @brief Search op on the consumer chain (bypass view like operations), until target operation is found or reach the
/// last consumer.
/// @return mlir::Operation* if target op is found, otherwise return mlir::failure().
/// @param op The operation to start searching from
/// @param isTargetOpFound Predicate function to identify target operations
mlir::FailureOr<mlir::Operation*> searchOpConsumers(mlir::Operation* op,
                                                    const std::function<bool(mlir::Operation*)>& isTargetOpFound);
}  // namespace IE
}  // namespace vpux
