//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

//
// Generated
//

#include <vpux/compiler/dialect/config/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/config/attributes.hpp.inc>

namespace vpux {
namespace config {

//
// Resource kind value getter
//

template <typename ConcreteKind, typename ResourceOp>
ConcreteKind getKindValue(ResourceOp op) {
    VPUX_THROW_WHEN(!op.getKind(), "Can't find attributes for Operation");
    const auto maybeKind = vpux::config::symbolizeEnum<ConcreteKind>(op.getKind());
    VPUX_THROW_WHEN(!maybeKind.has_value(), "Unsupported attribute kind");
    return maybeKind.value();
}

//
// CompilationMode
//

void setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode);
bool hasCompilationMode(mlir::ModuleOp module);
CompilationMode getCompilationMode(mlir::Operation* op);

}  // namespace config
}  // namespace vpux
