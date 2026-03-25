//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/StringRef.h>
#include <string>

namespace vpux::Core {

constexpr llvm::StringLiteral NPU_MODULE_NAME = "NPUModule";

// Default mode: entryPoint's callee functions are nested in separate submodules (entryPoint is not nested)
// EntryPoint mode: entryPoint function with all its callee functions are nested in one `NPUModule` submodule
enum class NestingMode { Default, EntryPoint };

NestingMode parseNestingMode(std::string& mode);

}  // namespace vpux::Core
