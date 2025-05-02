//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isDepthwiseOp(mlir::Operation* op) {
    return mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(op);
}
