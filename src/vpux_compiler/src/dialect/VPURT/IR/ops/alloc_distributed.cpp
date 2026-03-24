//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

using namespace vpux;

void vpux::VPURT::AllocDistributed::getEffects(SmallVectorImpl<MemoryEffect>& effects) {
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), ::llvm::cast<::mlir::OpResult>(getBuffer()));
}
