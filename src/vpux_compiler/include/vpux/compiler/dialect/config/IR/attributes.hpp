//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpux/compiler/dialect/config/version.hpp>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <optional>

//
// Generated
//

#include <vpux/compiler/dialect/config/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/config/attributes.hpp.inc>

namespace vpux {
namespace config {

std::optional<config::Platform> getPlatform(mlir::Operation* op);
config::ArchKind getArch(config::Platform platform);

//
// CompilationMode
//

void setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode);
bool hasCompilationMode(mlir::ModuleOp module);
CompilationMode getCompilationMode(mlir::Operation* op);

//
// ELF ABI Version
//

void setElfAbiVersion(mlir::ModuleOp module, const Version& version);
std::optional<Version> getElfAbiVersion(mlir::Operation* op);

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
// CompileMethodDebatch
//

void setCompileMethodDebatch(mlir::ModuleOp module);
bool hasCompileMethodDebatch(mlir::ModuleOp module);

//
// PureHostCompileFunc
//

// `PureHostCompileFunc` attribute is used to mark a host function that contains only operations from MLIR
// dialects (tensor, scf, async, memref and arith) and which code is compiled into CPU code
void setPureHostCompileFuncAttribute(mlir::func::FuncOp func);
bool isPureHostCompileFunc(mlir::func::FuncOp func);

}  // namespace config
}  // namespace vpux
