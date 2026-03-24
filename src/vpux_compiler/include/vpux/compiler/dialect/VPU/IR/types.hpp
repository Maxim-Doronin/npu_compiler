//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/type_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux::VPU {
enum class MemoryKind : uint64_t;
class DistributionInfoAttr;
class SEAttr;
class SparsityCompressionAttr;
}  // namespace vpux::VPU

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPU/types.hpp.inc>
#undef GET_TYPEDEF_CLASSES
