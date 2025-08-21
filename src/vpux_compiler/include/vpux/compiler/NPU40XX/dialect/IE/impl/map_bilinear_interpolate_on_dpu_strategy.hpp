//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/map_bilinear_interpolate_on_dpu_strategy.hpp"

namespace vpux::IE::arch40xx {
class MapBilinearInterpolateOnDPUStrategy final : public vpux::IE::IMapBilinearInterpolateOnDPUStrategy {
public:
    MapBilinearInterpolateOnDPUStrategy(const bool interpolateAsSEOpInStrategy)
            : IMapBilinearInterpolateOnDPUStrategy(interpolateAsSEOpInStrategy) {
    }
    void prepareInterpolate(mlir::ConversionTarget& target, LogCb logCb) const override;
};

}  // namespace vpux::IE::arch40xx
