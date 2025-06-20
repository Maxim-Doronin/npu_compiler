//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/adaptive_stripping_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>

using namespace vpux;

bool hasAdaptiveStrippingOption(mlir::ModuleOp module, StringRef option) {
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find PipelineOptions to fetch adaptive stripping option");
        return false;
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(option);
    if (attrValue == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find config.OptionOp to fetch adaptive stripping option");
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

bool VPU::hasEnableAdaptiveStripping(mlir::ModuleOp module) {
    return hasAdaptiveStrippingOption(module, ENABLE_ADAPTIVE_STRIPPING);
}
