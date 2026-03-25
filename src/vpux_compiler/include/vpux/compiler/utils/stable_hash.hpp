//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Types.h>

namespace vpux {

//! @brief Returns a "stable" (always produces the same value) hash.
llvm::hash_code getStableHash(mlir::Type type);

}  // namespace vpux
