//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attr_interfaces.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>

void setCompileMethodDebatch(mlir::ModuleOp module);
bool hasCompileMethodDebatch(mlir::ModuleOp module);

//
// Generated
//

#include <vpux/compiler/dialect/IE/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/IE/attributes.hpp.inc>
