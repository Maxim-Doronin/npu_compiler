//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/ELFNPU37XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/const/ops.hpp.inc>
