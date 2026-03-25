//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Types.h>

llvm::raw_ostream& operator<<(llvm::raw_ostream& o, const vpux::VPURT::BufferSection& sec);

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUASM/types.hpp.inc>
#undef GET_TYPEDEF_CLASSES
