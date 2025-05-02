//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"

namespace vpux {
namespace IE {
namespace arch37xx {

bool isMixPrecisionSupported(mlir::Operation* origOp, bool isPReLUSupported, Logger log);
bool checkPostOp(IE::LayerWithPostOpInterface layerWithPostOp, bool isPerAxisQuantizedOutput, bool isFloatInput);

}  // namespace arch37xx
}  // namespace IE
}  // namespace vpux
