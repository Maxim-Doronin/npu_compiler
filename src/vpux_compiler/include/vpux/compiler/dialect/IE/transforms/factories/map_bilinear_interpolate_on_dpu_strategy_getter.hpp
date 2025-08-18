//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/map_bilinear_interpolate_on_dpu_strategy.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE {

std::unique_ptr<IMapBilinearInterpolateOnDPUStrategy> createMapBilinearInterpolateOnDPUStrategy(
        mlir::func::FuncOp funcOp, bool interpolateAsSEOpInStrategy);

}  // namespace vpux::IE
