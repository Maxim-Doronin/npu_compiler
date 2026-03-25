//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/dilated_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

namespace vpux::VPU {

bool isSEPDWConv(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<VPU::NCEDepthConvolutionOp>(op)) {
        return false;
    }
    const auto sparseInputTensor = mlir::dyn_cast<VPU::SparseTensorType>(op->getOperand(0).getType());
    if (sparseInputTensor == nullptr) {
        return false;
    }
    auto seAttr = sparseInputTensor.getSeAttr();
    return seAttr != nullptr;
}

}  // namespace vpux::VPU
