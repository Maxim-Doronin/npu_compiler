//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::Const {

mlir::FailureOr<SmallVector<int64_t>> getConstArrValue(mlir::Value input);
mlir::FailureOr<int64_t> getConstOrAttrValue(mlir::Value input, mlir::IntegerAttr attr);
mlir::FailureOr<SmallVector<int64_t>> getConstOrArrAttrValue(mlir::Value input, mlir::ArrayAttr attr);

}  // namespace vpux::Const
