//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

namespace vpux {
namespace VPURegMapped {

//
// Passes
//

std::unique_ptr<mlir::Pass> createDeduceDynamicMappedInferenceVersionPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace VPURegMapped
}  // namespace vpux
