//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {

/// @brief Checks if the ConvertOp is supported on DMA
/// @param convertOp template argument
/// @return boolean

template <typename T>
bool isConvertSupportedOnDMA(T convertOp) {
    auto module = convertOp.getOperation();
    // ConvertSWLayers2VPUIPSWKernelPass still rely on arch check logic here
    // Remove arch check when one-shot enabled, TODO: E#113196
    auto arch = config::getArch(module);
    if (arch < config::ArchKind::NPU40XX) {
        // Feature is only tested on 40XX+
        return false;
    }

    auto inputElementType = mlir::cast<vpux::NDTypeInterface>(convertOp.getInput().getType()).getElementType();
    auto outputElementType = convertOp.getDstElemType();

    return inputElementType.isF32() && (outputElementType.isBF16() || outputElementType.isF16());
}
}  // namespace vpux
