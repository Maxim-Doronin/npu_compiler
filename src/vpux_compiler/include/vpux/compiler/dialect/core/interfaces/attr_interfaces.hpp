//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <cassert>

namespace vpux::Core {

class InlinerDispatchAttrInterface;

/// @brief Adds the attribute {"inliner_dispatch": attr} to all operations with a CallOpInterface or
/// CallableOpInterface. This sets up the operations for the desired inliner semantics. Nested ModuleOps are ignored.
void setInlinerDispatchAttr(mlir::ModuleOp moduleOp, InlinerDispatchAttrInterface attr);

}  // namespace vpux::Core

//
// Generated
//

#include <vpux/compiler/dialect/core/attr_interfaces.hpp.inc>
