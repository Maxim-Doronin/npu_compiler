//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/static_shape_op_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>

using namespace vpux;

bool hasEnableExtraStaticShapeOpOption(mlir::ModuleOp module, StringRef option) {
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find PipelineOptions to fetch extra shape bound option");
        return false;
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(option);
    if (attrValue == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find config.OptionOp to fetch extra shape bound option");
        return false;
    }
    auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attrValue.getOptionValue());
    if (boolAttr == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to cast config.OptionOp to BoolAttr");
        return false;
    }
    return boolAttr.getValue();
}

bool VPU::hasEnableExtraStaticShapeOps(mlir::ModuleOp module) {
    return hasEnableExtraStaticShapeOpOption(module, ENABLE_EXTRA_STATIC_SHAPE_OPS);
}
