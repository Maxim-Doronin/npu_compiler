//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/bytecode/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/bytecode/IR/types.hpp"

#include <mlir/IR/BuiltinTypes.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/section.hpp.inc>

namespace vpux::bytecode {

constexpr auto FUNCTION_SECTION_NAME = "func_section";
constexpr auto CONSTANT_SECTION_NAME = "constant_section";
constexpr auto KERNEL_SECTION_NAME = "kernel_section";
constexpr auto STRING_SECTION_NAME = "string_section";
constexpr auto TYPE_SECTION_NAME = "type_section";

}  // namespace vpux::bytecode
