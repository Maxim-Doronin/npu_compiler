//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"

namespace vpux {
namespace IE {

bool isSoftMaxAxisInLastMemDim(IE::SoftMaxOp op);

}  // namespace IE
}  // namespace vpux
