//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/utils/dump_intermediate_values.hpp"
#include "vpux/compiler/dialect/core/utils/nesting_utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

namespace vpux::Core {

//
// Passes
//

std::unique_ptr<mlir::Pass> createMoveDeclarationsToTopPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPrintDotPass(StringRef fileName = {}, StringRef startAfter = {},
                                               StringRef stopBefore = {}, bool printOnlyDotInterFaces = false,
                                               bool printConst = false, bool printDeclarations = false);
std::unique_ptr<mlir::Pass> createDumpIntermediateValuesPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createDumpIntermediateValuesPass(StringRef configFileName,
                                                             const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createPackNestedModulesPass(Logger log = Logger::global(),
                                                        Core::NestingMode nestingMode = Core::NestingMode::Default,
                                                        bool enableProfiling = false);
std::unique_ptr<mlir::Pass> createUnpackNestedModulesPass(const Logger& log = Logger::global(),
                                                          Core::NestingMode nestingMode = Core::NestingMode::Default);

//
// Registration
//

void registerPasses();

}  // namespace vpux::Core
