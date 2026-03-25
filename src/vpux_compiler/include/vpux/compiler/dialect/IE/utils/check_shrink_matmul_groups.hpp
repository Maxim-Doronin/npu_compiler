//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"

namespace vpux {
namespace IE {

bool checkMatMul(IE::MatMulOp origOp);

bool checkTranspose(IE::TransposeOp transposeOp);

bool checkAffineReshape(IE::AffineReshapeOp affineReshapeOp);

bool checkBroadCast(IE::BroadcastOp broadcastOp);

bool shouldShrinkMatmulGroups(IE::MatMulOp matmulOp);
}  // namespace IE
}  // namespace vpux
