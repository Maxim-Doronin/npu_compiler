//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/map_bilinear_interpolate_on_dpu_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/map_bilinear_interpolate_on_dpu_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/impl/map_bilinear_interpolate_on_dpu_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::IE {

std::unique_ptr<IMapBilinearInterpolateOnDPUStrategy> createMapBilinearInterpolateOnDPUStrategy(
        mlir::func::FuncOp funcOp, const bool interpolateAsSEOpInStrategy) {
    const auto arch = config::getArch(funcOp);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<arch37xx::MapBilinearInterpolateOnDPUStrategy>(interpolateAsSEOpInStrategy);
    default:
        return std::make_unique<arch40xx::MapBilinearInterpolateOnDPUStrategy>(interpolateAsSEOpInStrategy);
    }
    VPUX_THROW("Unable to get MapBilinearInterpolateOnDPUStrategy for arch {0}", arch);
}
}  // namespace vpux::IE
