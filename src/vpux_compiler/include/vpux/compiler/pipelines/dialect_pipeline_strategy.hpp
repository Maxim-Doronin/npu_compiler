//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {

//
// This is dialect-wise pipeline strategy
// Each platform should specify its own sub-pipeline implementation
//

class IDialectPipelineStrategy {
public:
    virtual void initializePipeline(mlir::OpPassManager&, Logger) {
        VPUX_THROW("Not implemented!");
    }

    virtual void buildIEPipeline(mlir::OpPassManager&, Logger) {
        VPUX_THROW("Not implemented!");
    }
    virtual void buildLowerIE2VPUPipeline(mlir::OpPassManager&, Logger) {
        VPUX_THROW("Not implemented!");
    }
    virtual void buildVPUPipeline(mlir::OpPassManager&, Logger) {
        VPUX_THROW("Not implemented!");
    }
    virtual void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager&, Logger) {
        VPUX_THROW("Not implemented!");
    }
    virtual void buildVPUIPPipeline(mlir::OpPassManager&, Logger) {
        VPUX_THROW("Not implemented!");
    }

    virtual ~IDialectPipelineStrategy() = default;
};

}  // namespace vpux
