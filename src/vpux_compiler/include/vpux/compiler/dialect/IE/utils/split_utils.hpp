//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<vpux::Dim> getSplitDimToShape1(IE::SplitOp splitOp);

}  // namespace IE
}  // namespace vpux
