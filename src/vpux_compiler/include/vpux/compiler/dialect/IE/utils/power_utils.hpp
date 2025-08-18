//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"

namespace vpux {
namespace IE {

std::optional<float> getExponentSplatVal(IE::PowerOp powerOp);

}  // namespace IE
}  // namespace vpux
