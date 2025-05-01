//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/static_shape_op_utils.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool hasEnableExtraStaticShapeOpOption(mlir::ModuleOp module, StringRef option) {
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find PipelineOptions to fetch extra shape bound option");
        return false;
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(option);
    if (attrValue == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find IE.OptionOp to fetch extra shape bound option");
        return false;
    }
    return static_cast<bool>(attrValue.getOptionValue());
}

bool VPU::hasEnableExtraStaticShapeOps(mlir::ModuleOp module) {
    return hasEnableExtraStaticShapeOpOption(module, ENABLE_EXTRA_STATIC_SHAPE_OPS);
}
