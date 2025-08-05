//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sep_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>

using namespace vpux;

bool hasSEOption(mlir::ModuleOp module, StringRef seOption) {
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find PipelineOptions to fetch SEP option");
        return false;
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(seOption);
    if (attrValue == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find config.OptionOp to fetch SEP option");
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

bool VPU::hasEnableSEPtrsOperations(mlir::ModuleOp module) {
    return hasSEOption(module, ENABLE_SE_PTRS_OPERATIONS);
}

bool VPU::hasEnableExperimentalSEPtrsOperations(mlir::ModuleOp module) {
    return hasSEOption(module, ENABLE_EXPERIMENTAL_SE_PTRS_OPERATIONS);
}
