//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

//
// Dialect hooks
//

void config::ConfigDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/config/attributes.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/config/enums.cpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/config/attributes.cpp.inc>

//
// CompilationMode
//

namespace {

constexpr StringLiteral compilationModeAttrName = "config.compilationMode";

}  // namespace

void vpux::config::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    module->setAttr(compilationModeAttrName, config::CompilationModeAttr::get(module.getContext(), compilationMode));
}

bool vpux::config::hasCompilationMode(mlir::ModuleOp module) {
    return module->hasAttr(compilationModeAttrName);
}

config::CompilationMode vpux::config::getCompilationMode(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (auto attr = module->getAttr(compilationModeAttrName)) {
        VPUX_THROW_UNLESS(mlir::isa<vpux::config::CompilationModeAttr>(attr),
                          "Module attribute '{0}' has unsupported value '{1}'", compilationModeAttrName, attr);

        return mlir::cast<vpux::config::CompilationModeAttr>(attr).getValue();
    }

    // Use DefaultHW as a default mode
    return config::CompilationMode::DefaultHW;
}
