//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/convert_to_palletization_lut_strategy_getter.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/impl/convert_to_palletization_lut_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux::IE {

std::unique_ptr<IConversionPassStrategy> createConvertToPalletizationLUTStrategy(mlir::func::FuncOp funcOp) {
    const auto arch = VPU::getArch(funcOp);
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX: {
        return std::make_unique<arch40xx::ConvertToPalletizationLUTStrategy>();
    }
    default: {
    }
    }
    VPUX_THROW("Unable to get ConvertToPalletizationLUTStrategy for arch {0}", arch);
}

}  // namespace vpux::IE
