//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// IPipelineRegistry
//

class IPipelineRegistry {
public:
    virtual void registerPipelines() = 0;

    virtual ~IPipelineRegistry() = default;
};

//
// createPipelineRegistry
//

std::unique_ptr<IPipelineRegistry> createPipelineRegistry(config::ArchKind archKind);

}  // namespace vpux
