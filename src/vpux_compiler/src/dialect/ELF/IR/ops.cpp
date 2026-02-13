//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"
#include "vpux/utils/core/optional.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/Builders.h>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/ELF/ops.cpp.inc>
