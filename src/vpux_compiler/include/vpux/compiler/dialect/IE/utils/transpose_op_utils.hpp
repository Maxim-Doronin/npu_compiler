//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"

namespace vpux {
namespace IE {

DimsOrder deduceInverseOrder(IE::TransposeOp op);

bool isWHSwappingTranspose(IE::TransposeOp op);

}  // namespace IE
}  // namespace vpux
