//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/compiler.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <transformations/utils/utils.hpp>

namespace vpux {
namespace ELFNPU37XX {

std::vector<uint8_t> exportToELF(mlir::ModuleOp module, Logger log = Logger::global());
BlobView exportToELF(mlir::ModuleOp module, BlobAllocator& allocator, Logger log = Logger::global());

}  // namespace ELFNPU37XX
}  // namespace vpux
