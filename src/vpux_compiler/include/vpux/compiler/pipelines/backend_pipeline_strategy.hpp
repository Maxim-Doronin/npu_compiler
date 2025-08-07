//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {

//
// This strategy is responsible for building a "backend" pipeline.
//

class IBackendPipelineStrategy {
public:
    virtual void buildELFPipeline(mlir::OpPassManager& pm, const intel_npu::Config& config,
                                  mlir::TimingScope& rootTiming, Logger log, bool useWlm) = 0;

    virtual ~IBackendPipelineStrategy() = default;
};

}  // namespace vpux
