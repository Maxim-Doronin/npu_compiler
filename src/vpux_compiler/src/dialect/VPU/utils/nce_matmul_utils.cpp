//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace vpux::VPU {

mlir::RankedTensorType inferNCEMatmulOutputType(vpux::NDTypeInterface input1Type, vpux::NDTypeInterface input2Type,
                                                vpux::NDTypeInterface origOutputType) {
    const auto input1Shape = input1Type.getShape();
    const auto input2Shape = input2Type.getShape();
    SmallVector<int64_t> outputShape{input1Shape[Dim(0)], input1Shape[Dim(1)], input2Shape[Dim(1)], input1Shape[Dim(3)],
                                     input1Shape[Dim(4)]};

    return mlir::RankedTensorType::get(outputShape, origOutputType.getElementType(),
                                       VPU::createTensorAttrFromType(input1Type));
}

bool isNCEMatMulSupported(vpux::NDTypeInterface inputType, [[maybe_unused]] vpux::NDTypeInterface filterType,
                          vpux::NDTypeInterface outputType, mlir::ModuleOp moduleOp, vpux::LogCb logCb,
                          bool checkLayout, [[maybe_unused]] bool checkChannelAlignment) {
    if (auto inOrder = inputType.getDimsOrder(); checkLayout && inOrder != DimsOrder::GNHWC) {
        logCb(llvm::formatv("VPU::NCEMatMulOp input has unsupported layout '{0}'", inOrder));
        return false;
    }

    // If we have less groups than clusters, it doesn't make sense to try split-over-group optimisation.
    const auto groups = outputType.getShape()[DimsGroups5D::Act::G];
    const auto clusters = config::getTileExecutor(moduleOp).getCount();

    if (groups < clusters) {
        logCb(llvm::formatv("VPU::NCEMatMulOp input has fewer groups than there are available clusters"));
        return false;
    }

    return true;
}

}  // namespace vpux::VPU
