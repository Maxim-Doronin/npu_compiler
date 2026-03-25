//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace mlir {
class Operation;
}  // namespace mlir

namespace vpux::ShaveCodeGen {

bool hasOnlySupportedTypes(mlir::Operation* op);

}  // namespace vpux::ShaveCodeGen
