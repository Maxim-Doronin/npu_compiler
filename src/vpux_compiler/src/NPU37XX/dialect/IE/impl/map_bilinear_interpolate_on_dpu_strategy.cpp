//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/map_bilinear_interpolate_on_dpu_strategy.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/map_bilinear_interpolate_on_DPU.hpp"

namespace vpux::IE::arch37xx {

void MapBilinearInterpolateOnDPUStrategy::prepareInterpolate(mlir::ConversionTarget& target, LogCb logCb) const {
    target.addDynamicallyLegalOp<IE::InterpolateOp>([this, logCb](IE::InterpolateOp op) {
        // For interpolation on axes H & W, and C <= 4,
        // SW kernel performance is bigger that DPU decomposition performance for floating scale factors
        const auto inputShape = getShape(op.getInput());
        if (inputShape.size() != 4 || inputShape[Dims4D::Act::C] > 4) {
            return isLegalInterpolateOp(op, _interpolateAsSEOpInStrategy, logCb);
        }

        const auto outputShape = getShape(op.getOutput());
        const auto attr = op.getAttr();
        const auto coordModeAttr = attr.getCoordMode();
        bool isAlignCorners = coordModeAttr.getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS;
        auto isIntegerRatio = [&](const auto& dim) -> bool {
            auto outputDim = outputShape[dim];
            auto inputDim = inputShape[dim];

            if (isAlignCorners) {
                outputDim = outputDim == 1 ? 1 : (outputDim - 1);
                inputDim = inputDim == 1 ? 1 : (inputDim - 1);
            }

            return (outputDim % inputDim == 0) || (inputDim % outputDim == 0);
        };

        const bool isInterpOnHW = inputShape[Dims4D::Act::N] == 1 && outputShape[Dims4D::Act::N] == 1 &&
                                  inputShape[Dims4D::Act::H] != outputShape[Dims4D::Act::H] &&
                                  inputShape[Dims4D::Act::W] != outputShape[Dims4D::Act::W] &&
                                  inputShape[Dims4D::Act::C] == outputShape[Dims4D::Act::C];

        if (isInterpOnHW && !isIntegerRatio(Dims4D::Act::H) && !isIntegerRatio(Dims4D::Act::W)) {
            return true;
        }
        return isLegalInterpolateOp(op, _interpolateAsSEOpInStrategy, logCb);
    });
}
}  // namespace vpux::IE::arch37xx
