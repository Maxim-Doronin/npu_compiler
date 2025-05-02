//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Dialect.h>

#pragma once

namespace vpux {

namespace arch40xx {

//
// registerBufferizableOpInterfaces
//

void registerBufferizableOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace arch40xx

}  // namespace vpux
