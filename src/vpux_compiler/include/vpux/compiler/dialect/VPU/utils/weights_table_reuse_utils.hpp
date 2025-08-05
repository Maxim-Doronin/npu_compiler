//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {
constexpr StringRef WEIGHTS_TABLE_REUSE_MODE = "VPU.WeightsTableReuseMode";

WeightsTableReuseMode getWeightsTableReuseMode(mlir::Operation* op);
bool isWeightsTableReuseEnabled(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
