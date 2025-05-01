//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "mlir/Interfaces/TilingInterface.h"

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_general_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_scf_algorithm.hpp"

using namespace vpux;
using namespace VPU;

TilingContext::TilingContext(mlir::Operation* operation): _operation(operation) {
}

void TilingContext::setTiling(std::unique_ptr<ITilingAlgorithm> tilingAlgorithm) {
    _tilingAlgorithm = std::move(tilingAlgorithm);
}

mlir::LogicalResult TilingContext::applyTiling(mlir::RewriterBase& builder, Logger log) {
    VPUX_THROW_WHEN(_tilingAlgorithm == nullptr, "Tiling algorithm is not specified");

    return _tilingAlgorithm->applyTiling(_operation, builder, log);
}

bool isSCFSupported(mlir::Operation* operation, ShapeRef tilingStrategy) {
    // E-162801 extend for operations with > 1 output
    if (operation->getNumResults() > 1) {
        return false;
    }

    const auto outShape = getShape(operation->getResult(0));

    // E-162627 extend to dynamic shapes
    if (outShape.isDynamic()) {
        return false;
    }

    // E-162801 extend to multi axes tiling
    auto tilingDims = getNonOneDim(tilingStrategy);
    if (tilingDims.size() != 1) {
        return false;
    }
    return true;
}

TilingContext vpux::VPU::createTilingContext(mlir::Operation* operation, ShapeRef strategy, bool enableSCFTiling) {
    TilingContext context(operation);

    std::unique_ptr<ITilingAlgorithm> algorithm;

    if (enableSCFTiling && mlir::isa<mlir::TilingInterface>(operation) && isSCFSupported(operation, strategy)) {
        algorithm = std::make_unique<TilingSCFAlgorithm>();
    } else {
        algorithm = std::make_unique<TilingGeneralAlgorithm>();
    }

    context.setTiling(std::move(algorithm));

    return context;
}
