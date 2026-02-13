//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Operation.h>
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/utils/logger/logger.hpp"

#pragma once

namespace vpux {
namespace VPU {

bool checkForQuantization(mlir::Operation* op, mlir::Operation* postOp);
bool hasPerChannelQuantizedOutput(mlir::Operation* op);

template <typename ConcreteModel, typename MainOpType>
class LayerWithPostOpModelBase : public IE::LayerWithPostOpInterface::ExternalModel<ConcreteModel, MainOpType> {
public:
    bool isSupportedClampOp(mlir::Operation* mainOp, mlir::Operation* maybeClampOp, const LogCb& logCb) const {
        auto clampOp = mlir::cast<vpux::IE::ClampOp>(maybeClampOp);

        if (config::getCompilationMode(clampOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!ConcreteModel::isSupportedHWClampOp(mainOp, clampOp, logCb)) {
            return false;
        }

        return VPU::NCEInvariant::isSupported(mlir::cast<MainOpType>(mainOp)).succeeded();
    }

    bool isSupportedPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) const {
        if (config::getCompilationMode(postOp) == config::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!ConcreteModel::isSupportedHWPostOp(mainOp, postOp, logCb)) {
            return false;
        }

        return VPU::NCEInvariant::isSupported(mlir::cast<MainOpType>(mainOp)).succeeded();
    }
};

}  // namespace VPU
}  // namespace vpux
