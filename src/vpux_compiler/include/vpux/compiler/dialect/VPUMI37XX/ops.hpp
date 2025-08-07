//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpux_headers/metadata.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <mlir/IR/BuiltinOps.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUMI37XX/ops.hpp.inc>
