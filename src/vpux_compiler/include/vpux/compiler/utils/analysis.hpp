//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux::VPUIP {
class LayerOpInterface;
}

namespace vpux {

//
// getFirstUser
//

mlir::Operation* getFirstUser(mlir::Value output);

//
// hasOneUniqueUser
//

bool hasOneUniqueUser(mlir::Operation* op);

//
// isBufAllocOp
//

bool isBufAllocOp(mlir::Operation* op);

//
// getInputsSanitized
//

mlir::SmallVector<mlir::Value> getInputsSanitized(VPUIP::LayerOpInterface layerOp);

//
// getModuleOp
//

mlir::ModuleOp getModuleOp(mlir::Operation* op);

/// @brief This function returns the top level ModuleOp which contains this op. If the op itself is the top level
/// ModuleOp, then op is returned. If it doesn't exist, an exception is thrown.
mlir::ModuleOp getTopModuleOp(mlir::Operation* op);

}  // namespace vpux
