//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "vpux/compiler/pipelines/dialect_pipeline_strategy.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {

//
// Within a frontend pipeline strategy, we may want to use different options,
// i.e. different dialect strategies. This is why we need a factory method.
// So for example "CustomMode" here can setup different option values
// for the IE dialect pipeline than "DefaultHW" mode:
//
//     auto strategy = _createPipelineStrategy(config::CompilationMode::DefaultHW);
//     strategy->buildIEPipeline(pm, _log);
//     ..
//     auto& nestedPm = pm.nest<mlir::ModuleOp>();
//     auto nestedStrategy = _createPipelineStrategy(config::CompilationMode::CustomMode);
//     nestedStrategy->buildIEPipeline(nestedPm, _log);
//

using StrategyFactoryFn = std::function<std::unique_ptr<IDialectPipelineStrategy>(config::CompilationMode)>;

//
// This factory is responsible for building a "frontend" pipeline.
// For example, default pipeline: IE -> VPU -> VPUIP
//

class IFrontendPipelineStrategy {
public:
    IFrontendPipelineStrategy(StrategyFactoryFn createPipelineStrategy, Logger log)
            : _createPipelineStrategy(std::move(createPipelineStrategy)), _log(log) {
    }

    virtual void buildPipeline(mlir::OpPassManager& pm) = 0;

    virtual ~IFrontendPipelineStrategy() = default;

protected:
    StrategyFactoryFn _createPipelineStrategy;
    Logger _log;
};

}  // namespace vpux
