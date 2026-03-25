//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {

class IMapBilinearInterpolateOnDPUStrategy {
public:
    IMapBilinearInterpolateOnDPUStrategy(const bool interpolateAsSEOp)
            : _interpolateAsSEOpInStrategy(interpolateAsSEOp) {
    }
    virtual void prepareInterpolate(mlir::ConversionTarget& target, LogCb logCb) const = 0;

    virtual ~IMapBilinearInterpolateOnDPUStrategy() = default;

protected:
    bool _interpolateAsSEOpInStrategy = false;
};

}  // namespace vpux::IE
